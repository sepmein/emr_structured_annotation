import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.benchmark_model_service import (
    RequestFailure, benchmark_service, main, make_transport, parse_prediction, read_targets,
)
from scripts.evaluate_entity_predictions import build_report, normalize_records


GROUPS = {"symptons_labels": ["发热"]}
XML = '<View><Labels name="symptons_labels" toName="chief_complaint_text"><Label value="发热"/></Labels><Text name="chief_complaint_text" value="$text"/></View>'
TASKS = [{"task_id": "positive", "text": "无发热。", "entities": "gold must not be read", "patient_id": "private-id"},
         {"task_id": "negative", "text": "一般情况可。", "entities": []}]


def response(entities=True, version="test-model"):
    result = [{"type": "labels", "from_name": "symptons_labels", "to_name": "chief_complaint_text",
               "value": {"start": 1, "end": 3, "text": "发热", "labels": ["发热"]}}] if entities else []
    return {"results": [{"result": result, "model_version": version}]}


class BenchmarkModelServiceTests(unittest.TestCase):
    def test_gold_metadata_not_sent_warmup_excluded_and_first_predictions_retained(self):
        calls = []
        def transport(payload):
            calls.append(payload)
            positive = payload["tasks"][0]["id"] == "positive"
            return response(positive), 200
        predictions, report = benchmark_service(TASKS, GROUPS, XML, "explicit-project", transport, 3, 1)
        self.assertEqual(len(calls), 8)
        self.assertEqual(report["measured_task_runs"], 6)
        self.assertEqual(report["warmup_task_runs"], 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[1]["entities"], [])
        self.assertEqual(report["successful_task_runs"], 6)
        self.assertGreater(report["successful_tasks_per_second"], 0)
        self.assertGreaterEqual(report["p95_success_latency_ms"], report["p50_success_latency_ms"])
        self.assertTrue(all(set(c["tasks"][0]) == {"id", "data"} for c in calls))
        self.assertNotIn("private-id", json.dumps(calls))
        self.assertNotIn("一般情况可", json.dumps(report, ensure_ascii=False))
        self.assertIsNone(report["token_coverage"]["truncation_rate"])

    def test_failure_not_replaced_by_later_success_and_scoring_denominator_retained(self):
        calls = 0
        def transport(payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RequestFailure("timeout")
            return response(), 200
        predictions, report = benchmark_service(TASKS[:1], GROUPS, XML, "1", transport, 2, 0)
        self.assertEqual(predictions[0]["status"], "failed")
        self.assertEqual(predictions[0]["entities"], [])
        self.assertEqual(report["failure_rate"], .5)
        self.assertEqual(report["later_runs_differing_from_first_prediction"], 1)
        gold = normalize_records([{"task_id": "positive", "text": "无发热。", "entities": [{"start": 1,"end": 3,"label": "发热"}]}], {"发热"}, reference=True, source="gold")
        pred = normalize_records(predictions, {"发热"}, reference=False, source="model")
        score = build_report(gold, pred, GROUPS)["runs"]["before"]
        self.assertEqual(score["per_label"]["发热"]["fn"], 1)
        self.assertEqual(score["coverage"]["failed_prediction_tasks"], 1)

    def test_all_failures_have_null_success_latency_and_zero_throughput(self):
        def transport(payload):
            raise RequestFailure("http_error", 503)
        predictions, report = benchmark_service(TASKS, GROUPS, XML, "1", transport, 2, 1)
        self.assertTrue(all(p["status"] == "failed" for p in predictions))
        self.assertEqual(report["error_categories"], {"http_error": 4})
        self.assertEqual(report["warmup_failed_runs"], 2)
        self.assertIsNone(report["p50_success_latency_ms"])
        self.assertIsNone(report["p95_success_latency_ms"])
        self.assertEqual(report["successful_tasks_per_second"], 0)

    def test_structural_http_success_is_not_entity_success(self):
        bad = [None, {"results": []}, {"results": [{"result": None}]},
               {"results": [response()["results"][0], response()["results"][0]]},
               {"results": [[response()["results"][0], response()["results"][0]]]}]
        for source in (response(), response(), response(), response(), response()):
            bad.append(source)
        bad[-5]["results"][0]["result"][0]["from_name"] = "wrong-control"
        bad[-4]["results"][0]["result"][0]["to_name"] = "wrong-target"
        bad[-3]["results"][0]["result"][0]["value"]["end"] = 99
        bad[-2]["results"][0]["result"][0]["value"]["text"] = "wrong-span"
        bad[-1]["results"][0]["result"][0]["value"]["labels"] = ["unknown"]
        for payload in bad:
            with self.subTest(payload=payload):
                _, report = benchmark_service(TASKS[:1], GROUPS, XML, "1", lambda _: (payload, 200), 1, 0)
                self.assertEqual(report["error_categories"], {"invalid_entity_response": 1})

    def test_single_nested_prediction_allowed_duplicates_preserved_and_nonentities_not_scored(self):
        task = next(iter(normalize_records([{"task_id": "1", "text": "无发热。", "entities": []}], {"发热"}, reference=True, source="test").values()))
        pred = response()["results"][0]
        pred["result"].append(pred["result"][0])
        pred["result"].append({"type": "relation"})
        entities, version, ignored = parse_prediction({"results": [[pred]]}, task, GROUPS, read_targets(XML, GROUPS))
        self.assertEqual(len(entities), 2)
        self.assertEqual(ignored, 1)
        self.assertEqual(version, "test-model")

    def test_version_and_prediction_changes_detected(self):
        calls = 0
        def transport(payload):
            nonlocal calls
            calls += 1
            return response(calls == 1, f"version-{calls}"), 200
        predictions, report = benchmark_service(TASKS[:1], GROUPS, XML, "1", transport, 2, 0)
        self.assertEqual(len(predictions[0]["entities"]), 1)
        self.assertTrue(report["multiple_model_versions_seen"])
        self.assertEqual(report["later_runs_differing_from_first_prediction"], 1)

    def test_invalid_inputs_fail_before_transport(self):
        for payload, xml, repeats, warmup, project in (([], XML, 1, 0, "1"), (TASKS, XML, 0, 0, "1"),
                    (TASKS, XML, 1, -1, "1"), (TASKS, XML.replace("$text", "$other"), 1, 0, "1"), (TASKS, XML, 1, 0, "")):
            with self.subTest(payload=payload, repeats=repeats), self.assertRaises(ValueError):
                benchmark_service(payload, GROUPS, xml, project, lambda _: self.fail("Unexpected network call"), repeats, warmup)
        for endpoint, timeout, username, password in (("http://user:password@localhost/predict", 1, None, None),
            ("http://localhost/predict?token=secret", 1, None, None), ("file:///x", 1, None, None),
            ("http://localhost/predict", float("nan"), None, None), ("http://localhost/predict", 1, "user", None)):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                make_transport(endpoint, timeout, username, password)

    def test_http_transport_with_local_stub_and_safe_error_categories(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/predict")
                    self.end_headers()
                    return
                self.send_response(503 if self.path == "/error" else 200)
                self.end_headers()
                self.wfile.write(b'private-error-response' if self.path in {"/error", "/invalid-json"} else json.dumps(response()).encode("utf-8"))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}"
            payload, status = make_transport(endpoint + "/predict", 2, None, None)({"test": 1})
            self.assertEqual(status, 200)
            self.assertEqual(payload["results"][0]["model_version"], "test-model")
            for path, category, status in (("/error", "http_error", 503), ("/invalid-json", "invalid_json", 200), ("/redirect", "http_error", 302)):
                with self.subTest(path=path), self.assertRaises(RequestFailure) as caught:
                    make_transport(endpoint + path, 2, None, None)({"test": 1})
                self.assertEqual(caught.exception.category, category)
                self.assertEqual(caught.exception.http_status, status)
                self.assertNotIn("private-error-response", str(caught.exception))
            self.assertEqual(len(captured), 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_cli_writes_hashes_and_protects_existing_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks, config, out, bench = [root / name for name in ("tasks.json", "config.xml", "prediction.json", "timing.json")]
            tasks.write_text(json.dumps(TASKS[:1]), encoding="utf-8")
            config.write_text(XML, encoding="utf-8")
            args = ["--tasks", str(tasks), "--label-config", str(config), "--endpoint", "http://localhost:9090/predict", "--project", "1",
                    "--run-name", "before", "--model-artifact-id", "fictional-transport-test", "--output", str(out), "--benchmark", str(bench),
                    "--repeats", "1", "--warmup", "0", "--dataset-kind", "synthetic_demo"]
            with patch("scripts.benchmark_model_service.make_transport", return_value=lambda _: (response(), 200)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(bench.read_text(encoding="utf-8"))
            self.assertEqual(report["run_name"], "before")
            self.assertEqual(report["dataset_kind"], "synthetic_demo")
            self.assertEqual(len(report["inputs"]["tasks"]["sha256"]), 64)
            self.assertNotIn("localhost", bench.read_text(encoding="utf-8"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
