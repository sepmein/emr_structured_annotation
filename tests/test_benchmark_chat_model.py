import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.benchmark_chat_model import (
    ChatAdapter, make_chat_transport, make_prompt, main, parse_chat_response,
    reported_usage, run_chat_benchmark,
)
from scripts.benchmark_model_service import RequestFailure, read_targets
from scripts.compare_model_benchmarks import summarize
from scripts.evaluate_entity_predictions import build_report, normalize_records


GROUPS = {"symptons_labels": ["发热"], "time_labels": ["时间表达"]}
XML = '<View><Labels name="symptons_labels" toName="chief_complaint_text"><Label value="发热"/></Labels><Labels name="time_labels" toName="chief_complaint_text"><Label value="时间表达"/></Labels><Relations><Relation value="持续时长"/></Relations><Text name="chief_complaint_text" value="$text"/></View>'
TARGETS = read_targets(XML, GROUPS)
TASKS = [{"task_id": "positive", "text": "发热三天", "entities": "must never be read", "patient_id": "private-patient"},
         {"task_id": "negative", "text": "一般可。", "entities": []}]
SETTINGS = {"model": "fictional-chat-model", "stream": False, "n": 1, "max_tokens": 1024}


def extracted(positive=True):
    return {"entities": [{"id": "e1", "start": 0, "end": 2, "text": "发热", "label": "发热"},
                          {"id": "e2", "start": 2, "end": 4, "text": "三天", "label": "时间表达"}] if positive else [],
            "relations": [{"from_id": "e1", "to_id": "e2", "type": "持续时长", "direction": "right"}] if positive else []}


def response(content=None, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": json.dumps(extracted() if content is None else content, ensure_ascii=False)}}],
            "model": "fictional-chat-model-v1", "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens_details": {"reasoning_tokens": 5}}}


def adapter(transport):
    return ChatAdapter(transport, SETTINGS, make_prompt(GROUPS, ["持续时长"], "虚构规范"), GROUPS, TARGETS, ["持续时长"])


class ChatBenchmarkTests(unittest.TestCase):
    def test_correct_entities_relations_and_negative_return_explicit_results(self):
        parsed, content = parse_chat_response(response(), "发热三天", GROUPS, TARGETS, ["持续时长"])
        self.assertEqual(len(parsed["results"][0]["result"]), 3)
        self.assertEqual(content["relations"][0]["from_id"], "e1")
        self.assertEqual(parsed["results"][0]["result"][0]["from_name"], "symptons_labels")
        negative, _ = parse_chat_response(response(extracted(False)), "一般可。", GROUPS, TARGETS, ["持续时长"])
        self.assertEqual(negative["results"][0]["result"], [])

    def test_unknown_labels_bad_spans_text_and_dangling_relations_rejected(self):
        mutations = [lambda c: c["entities"][0].update(label="未知标签"), lambda c: c["entities"][0].update(start=True),
                     lambda c: c["entities"][0].update(end=99), lambda c: c["entities"][0].update(text="改写"),
                     lambda c: c["entities"][1].update(id="e1"), lambda c: c["relations"][0].update(to_id="missing"),
                     lambda c: c["relations"][0].update(direction="left"), lambda c: c.pop("relations")]
        for mutate in mutations:
            content = extracted()
            mutate(content)
            with self.subTest(content=content), self.assertRaises((ValueError, RequestFailure)):
                parse_chat_response(response(content), "发热三天", GROUPS, TARGETS, ["持续时长"])

    def test_non_bmp_offsets_use_exact_python_character_positions(self):
        content = {"entities": [{"id": "e1", "start": 1, "end": 3, "text": "发热", "label": "发热"}], "relations": []}
        parse_chat_response(response(content), "🧪发热", GROUPS, TARGETS, ["持续时长"])
        content["entities"][0].update(start=2, end=4)
        with self.assertRaises(ValueError):
            parse_chat_response(response(content), "🧪发热", GROUPS, TARGETS, ["持续时长"])

    def test_duplicate_spans_with_distinct_ids_preserved_for_false_positive_scoring(self):
        content = extracted()
        duplicate = {**content["entities"][0], "id": "e3"}
        content["entities"].append(duplicate)
        parsed, _ = parse_chat_response(response(content), "发热三天", GROUPS, TARGETS, ["持续时长"])
        self.assertEqual(len(parsed["results"][0]["result"]), 4)

    def test_truncation_refusal_tool_calls_and_multiple_choices_rejected(self):
        for finish in ("length", "content_filter", "tool_calls", None):
            with self.assertRaises(RequestFailure):
                parse_chat_response(response(finish=finish), "发热三天", GROUPS, TARGETS, ["持续时长"])
        for key, value in (("refusal", "fictional refusal"), ("tool_calls", [{"type": "function"}])):
            payload = response()
            payload["choices"][0]["message"][key] = value
            with self.assertRaises(RequestFailure):
                parse_chat_response(payload, "发热三天", GROUPS, TARGETS, ["持续时长"])
        payload = response()
        payload["choices"].append(copy.deepcopy(payload["choices"][0]))
        with self.assertRaises(ValueError):
            parse_chat_response(payload, "发热三天", GROUPS, TARGETS, ["持续时长"])

    def test_no_fence_removal_or_offset_repair(self):
        payload = response()
        payload["choices"][0]["message"]["content"] = '```json\n' + payload["choices"][0]["message"]["content"] + '\n```'
        with self.assertRaises(ValueError):
            parse_chat_response(payload, "发热三天", GROUPS, TARGETS, ["持续时长"])

    def test_requests_contain_only_guidance_schema_and_exact_text_no_reference(self):
        requests = []
        def transport(payload):
            requests.append(payload)
            text = json.loads(payload["messages"][1]["content"])["text"]
            return response(extracted(text == "发热三天")), 200
        predictions, report, structured = run_chat_benchmark(TASKS, GROUPS, XML, adapter(transport), 2, 1)
        self.assertEqual(len(requests), 6)
        self.assertEqual(report["warmup_task_runs"], 2)
        self.assertEqual(report["measured_task_runs"], 4)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(structured), 2)
        self.assertEqual(predictions[1]["entities"], [])
        transmitted = json.dumps(requests, ensure_ascii=False)
        for value in ("private-patient", "must never be read", '"positive"', '"negative"'):
            self.assertNotIn(value, transmitted)
        self.assertEqual(json.loads(requests[0]["messages"][1]["content"]), {"text": "发热三天", "text_chars": 4})
        self.assertEqual(report["samples"][0]["usage"]["cached_tokens"], 20)
        self.assertEqual(report["samples"][0]["usage"]["reasoning_tokens"], 5)
        self.assertIsNone(report["token_coverage"]["truncation_rate"])
        self.assertEqual(structured[0]["prediction"]["relations"][0]["type"], "持续时长")
        self.assertNotIn("发热三天", json.dumps(report, ensure_ascii=False))

    def test_first_token_limited_response_not_replaced_by_later_success(self):
        count = 0
        def transport(payload):
            nonlocal count
            count += 1
            return response(finish="length" if count == 1 else "stop"), 200
        predictions, report, structured = run_chat_benchmark(TASKS[:1], GROUPS, XML, adapter(transport), 2, 0)
        self.assertEqual(predictions[0]["status"], "failed")
        self.assertIsNone(structured[0]["prediction"])
        self.assertEqual(report["failure_rate"], .5)
        self.assertEqual(report["samples"][0]["error_category"], "output_token_limit")
        self.assertEqual(report["samples"][0]["usage"]["prompt_tokens"], 100)
        gold = normalize_records([{"task_id": "positive", "text": "发热三天", "entities": extracted()["entities"]}], {"发热", "时间表达"}, reference=True, source="gold")
        predicted = normalize_records(predictions, {"发热", "时间表达"}, reference=False, source="prediction")
        self.assertEqual(build_report(gold, predicted, GROUPS)["runs"]["before"]["per_label"]["发热"]["fn"], 1)

    def test_missing_or_invalid_usage_is_unknown_never_zero(self):
        self.assertTrue(all(v is None for v in reported_usage(None).values()))
        values = reported_usage({"usage": {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": 0}})
        self.assertIsNone(values["prompt_tokens"])
        self.assertIsNone(values["completion_tokens"])
        self.assertEqual(values["total_tokens"], 0)

    def test_http_transport_auth_redirect_and_safe_errors(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Authorization"), request))
                if self.path == "/redirect":
                    self.send_response(307)
                    self.send_header("Location", "/ok")
                    self.end_headers()
                    return
                self.send_response(200 if self.path != "/error" else 429)
                self.end_headers()
                self.wfile.write(json.dumps(response()).encode() if self.path == "/ok" else b'private response body')
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            payload, status = make_chat_transport(base + "/ok", 5, "fictional-token")({"model": "mock"})
            self.assertEqual(status, 200)
            self.assertEqual(seen[-1][1], "Bearer fictional-token")
            for path in ("/redirect", "/error", "/bad-json"):
                before = len(seen)
                with self.assertRaises(RequestFailure) as caught:
                    make_chat_transport(base + path, 5, None)({"model": "mock"})
                self.assertNotIn("private", str(caught.exception))
                self.assertEqual(len(seen), before + 1)
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                args = self.write_cli_inputs(folder)
                args[args.index("--endpoint") + 1] = base + "/ok"
                (folder / "tasks.json").write_text(json.dumps(TASKS[:1], ensure_ascii=False), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args + ["--no-auth"]), 0)
                report = json.loads((folder / "result/benchmark.json").read_text(encoding="utf-8"))
                self.assertEqual(summarize(report)["successful_requests"], 1)
                self.assertEqual(report["samples"][0]["usage"]["prompt_tokens"], 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def write_cli_inputs(self, folder):
        for name, text in (("tasks.json", json.dumps(TASKS, ensure_ascii=False)), ("config.xml", XML), ("guide.md", "虚构指导规范")):
            (folder / name).write_text(text, encoding="utf-8")
        return ["--tasks", str(folder / "tasks.json"), "--label-config", str(folder / "config.xml"), "--guidance", str(folder / "guide.md"),
                "--endpoint", "https://example.invalid/v1/chat/completions", "--model", "fictional-model", "--deployment", "external_api",
                "--max-output-tokens", "1024", "--dataset-kind", "synthetic_demo", "--output-dir", str(folder / "result")]

    def test_offline_cli_never_reads_key_or_calls_service_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.write_cli_inputs(folder)
            with patch("scripts.benchmark_chat_model.os.getenv", side_effect=AssertionError("must not read key")), patch("scripts.benchmark_chat_model.make_chat_transport", side_effect=AssertionError("must not create transport")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--check-only"]), 0)
            receipt = json.loads((folder / "result/run_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "offline_preflight_passed_no_requests")
            self.assertFalse((folder / "result/predictions.json").exists())
            self.assertNotIn("发热三天", (folder / "result/prompt.txt").read_text(encoding="utf-8"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--check-only"])

    def test_cli_mock_run_outputs_match_comparison_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.write_cli_inputs(folder)
            def transport(payload):
                text = json.loads(payload["messages"][1]["content"])["text"]
                return response(extracted(text == "发热三天")), 200
            with patch("scripts.benchmark_chat_model.make_chat_transport", return_value=transport), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--no-auth", "--json-mode", "--temperature", "0", "--token-limit-field", "max_completion_tokens"]), 0)
            report = json.loads((folder / "result/benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(summarize(report)["successful_requests"], 2)
            self.assertEqual(report["request_settings"]["max_completion_tokens"], 1024)
            self.assertEqual(report["request_settings"]["response_format"], {"type": "json_object"})
            self.assertEqual(report["inputs"]["tasks"]["file"], "tasks.json")
            self.assertEqual(report["token_coverage"]["status"], "usage_reported_when_available_full_text_processing_not_verified")

    def test_missing_key_or_invalid_task_fails_before_output_or_request(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.write_cli_inputs(folder)
            with patch.dict("os.environ", {}, clear=True), patch("scripts.benchmark_chat_model.make_chat_transport") as create, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            create.assert_not_called()
            self.assertFalse((folder / "result").exists())
            (folder / "tasks.json").write_text('[]', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--check-only"])


if __name__ == "__main__":
    unittest.main()
