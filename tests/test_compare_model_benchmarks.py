import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from emr_annotation.evaluation.compare_model_benchmarks import capacity_scenario, compare, main, read_context, summarize


def benchmark(scale=1, failed=()):
    samples = []
    for repeat in range(2):
        for index, chars in enumerate((100, 500, 2000)):
            samples.append({"task_id": str(index), "repeat": repeat, "phase": "measured", "text_chars": chars,
                            "latency_ms": scale * (10 * (index + 1) + repeat),
                            "status": "failed" if (index, repeat) in failed else "ok", "response_model_version": "mock"})
    return {"samples": samples, "repeats": 2, "concurrency": 1, "tasks_per_request": 1, "unique_tasks": 3,
            "measured_task_runs": 6, "measured_wall_seconds": sum(s["latency_ms"] for s in samples) / 1000 + .001,
            "measurement_scope": "serial_http_request_through_entity_validation", "dataset_kind": "synthetic_demo",
            "inputs": {"tasks": {"sha256": "a" * 64}}}


CONTEXT = {"ours": {"label": "专用模型（虚构）", "deployment": "local", "task_scope_id": "ner_re_v1", "resource_budget_id": "fictional-1-gpu"},
           "baseline": {"label": "对照模型（虚构）", "deployment": "local", "task_scope_id": "ner_re_v1", "resource_budget_id": "fictional-1-gpu"}}
SCENARIO = {"basis": "complete_records_per_second", "reviewed_quality_and_full_text": True,
            "sustained_rates": {"ours": 10, "baseline": 1}, "hours_per_day": 8, "planning_fraction": .7,
            "records_per_hospital_day": 20000, "peak_records_per_hospital_second": 1,
            "assumptions": "Entirely hypothetical; quality review and resource equality are simulated, not verified."}


class ModelBenchmarkComparisonTests(unittest.TestCase):
    def test_paired_statistics_keep_failures_and_unique_record_grain(self):
        ours = benchmark(failed=((2, 0), (2, 1)))
        ours.update(p50_success_latency_ms=999999, successful_tasks_per_second=999999)
        result = compare(ours, benchmark(2), CONTEXT)
        run = result["runs"]["ours"]
        self.assertEqual(run["unique_tasks"], 3)
        self.assertEqual(run["formal_requests"], 6)
        self.assertEqual(run["all_failed_tasks"], 1)
        self.assertEqual(run["failure_rate"], 2 / 6)
        self.assertEqual(run["p50_success_latency_ms"], 15.5)
        self.assertEqual(run["points"][0]["median_success_latency_ms"], 10.5)
        self.assertIsNone(run["points"][2]["median_success_latency_ms"])
        self.assertAlmostEqual(run["observed_successful_requests_per_second"], 4 / ours["measured_wall_seconds"])
        self.assertIsNone(result["capacity"])
        self.assertFalse(result["same_successful_request_population"])
        self.assertTrue(all(v is None for v in result["descriptive_latency_ratio_baseline_over_ours"].values()))

    def test_before_after_matching_requires_exact_common_source_and_tasks(self):
        for mutate in (lambda b: b["inputs"]["tasks"].update(sha256="b" * 64),
                       lambda b: b["samples"][0].update(task_id="other"),
                       lambda b: [s.update(text_chars=101) for s in b["samples"] if s["task_id"] == "0"]):
            baseline = benchmark()
            mutate(baseline)
            with self.assertRaises(ValueError):
                compare(benchmark(), baseline, CONTEXT)

    def test_duplicate_missing_repetition_and_warmup_cannot_inflate_results(self):
        for mutate in (lambda b: b["samples"].append(copy.deepcopy(b["samples"][0])),
                       lambda b: b["samples"].pop(),
                       lambda b: b["samples"][0].update(phase="warmup"),
                       lambda b: b.update(unique_tasks=1),
                       lambda b: b["samples"][0].update(repeat=2)):
            payload = benchmark()
            mutate(payload)
            with self.assertRaises(ValueError):
                summarize(payload)

    def test_invalid_numbers_and_serial_wall_time_rejected(self):
        for value in (True, float("nan"), float("inf"), -1):
            payload = benchmark()
            payload["samples"][0]["latency_ms"] = value
            with self.assertRaises(ValueError):
                summarize(payload)
        for wall in (0, .04):
            payload = benchmark()
            payload["measured_wall_seconds"] = wall
            with self.assertRaises(ValueError):
                summarize(payload)

    def test_all_failed_has_no_success_latency_or_ratio(self):
        payload = benchmark(failed=tuple((i, r) for i in range(3) for r in range(2)))
        result = compare(payload, benchmark(), CONTEXT)
        self.assertIsNone(result["runs"]["ours"]["p50_success_latency_ms"])
        self.assertEqual(result["runs"]["ours"]["observed_successful_requests_per_second"], 0)
        self.assertTrue(all(v is None for v in result["descriptive_latency_ratio_baseline_over_ours"].values()))

    def test_unaligned_measurement_displays_statistics_without_ratio(self):
        for key, value in (("measurement_scope", "server_compute_only"), ("concurrency", 2)):
            baseline = benchmark()
            baseline[key] = value
            result = compare(benchmark(), baseline, CONTEXT)
            self.assertFalse(result["aligned_latency_protocol_declared"])
            self.assertTrue(all(v is None for v in result["descriptive_latency_ratio_baseline_over_ours"].values()))
        self.assertEqual(compare(benchmark(), benchmark(2), CONTEXT)["descriptive_latency_ratio_baseline_over_ours"]["p50_success_latency_ms"], 2)

    def test_mixed_version_dataset_and_task_scope_rejected(self):
        baseline = benchmark()
        baseline["samples"][0]["response_model_version"] = "another-model"
        with self.assertRaises(ValueError):
            compare(benchmark(), baseline, CONTEXT)
        baseline = benchmark()
        baseline["dataset_kind"] = "human_test_set"
        with self.assertRaises(ValueError):
            compare(benchmark(), baseline, CONTEXT)
        context = copy.deepcopy(CONTEXT)
        context["baseline"]["task_scope_id"] = "ner_only"
        with self.assertRaises(ValueError):
            compare(benchmark(), benchmark(), context)

    def test_external_api_allowed_for_latency_but_not_same_compute_capacity(self):
        context = copy.deepcopy(CONTEXT)
        context["baseline"].update(deployment="external_api", resource_budget_id=None)
        self.assertIsNone(compare(benchmark(), benchmark(), context)["capacity"])
        with self.assertRaisesRegex(ValueError, "external API"):
            compare(benchmark(), benchmark(), context, SCENARIO)

    def test_capacity_uses_separate_rates_and_the_smaller_peak_limit(self):
        result = compare(benchmark(), benchmark(2), CONTEXT, SCENARIO)["capacity"]
        self.assertEqual(result["status"], "assumption_based_planning_only")
        self.assertEqual(result["models"]["ours"]["daily_planning_records"], 201600)
        self.assertEqual(result["models"]["ours"]["hospitals_by_daily_volume"], 10)
        self.assertEqual(result["models"]["ours"]["planning_hospitals"], 7)
        self.assertEqual(result["models"]["baseline"]["planning_hospitals"], 0)
        self.assertEqual(result["declared_sustained_rate_ratio"], 10)

    def test_missing_capacity_inputs_or_unequal_resources_rejected(self):
        for key, value in (("sustained_rates", None), ("reviewed_quality_and_full_text", False),
                           ("planning_fraction", 1.1), ("peak_records_per_hospital_second", 0), ("hours_per_day", 25)):
            scenario = copy.deepcopy(SCENARIO)
            scenario[key] = value
            with self.assertRaises(ValueError):
                capacity_scenario(scenario, read_context(CONTEXT), True)
        context = copy.deepcopy(CONTEXT)
        context["baseline"]["resource_budget_id"] = "different"
        with self.assertRaises(ValueError):
            capacity_scenario(SCENARIO, read_context(context), True)

    def test_cli_publishes_receipts_without_overwriting_or_partial_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name, payload in (("ours", benchmark()), ("baseline", benchmark(2)), ("context", CONTEXT)):
                (folder / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            args = ["--ours", str(folder / "ours.json"), "--baseline", str(folder / "baseline.json"),
                    "--context", str(folder / "context.json"), "--output-dir", str(folder / "result")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            receipt = json.loads((folder / "result/comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["sources"]["ours"]["sha256"]), 64)
            self.assertIn("虚构", (folder / "result/comparison.md").read_text(encoding="utf-8"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            args[-1] = str(folder / "plot-failed")
            with patch("scripts.compare_model_benchmarks.render_figure", side_effect=ImportError("plot dependency unavailable")), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--plot"])
            self.assertFalse((folder / "plot-failed").exists())


if __name__ == "__main__":
    unittest.main()
