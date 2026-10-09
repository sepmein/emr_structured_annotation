import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_entity_predictions import read_label_groups
from scripts.regex_entity_baseline import compile_rules, main, predict_entities, run_baseline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "label_studio/pneumonia_config.xml"
RULES = ROOT / "annotation_agent_workflow/rules/regex_entity_baseline_v0.1.0.json"


class RegexBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = read_label_groups(CONFIG)
        cls.payload = json.loads(RULES.read_text(encoding="utf-8"))
        cls.rules = compile_rules(cls.payload, cls.groups)

    def extract(self, text):
        return [(e["text"], e["label"]) for e in predict_entities(text, self.rules)]

    def test_all_current_entity_labels_have_explicit_rules(self):
        self.assertEqual(len(self.rules), 49)
        self.assertEqual({r.label for r in self.rules}, {v for values in self.groups.values() for v in values})

    def test_negated_mentions_retained_and_fever_not_inferred(self):
        self.assertIn(("发热", "发热"), self.extract("无发热。"))
        self.assertNotIn("发热", [label for _, label in self.extract("体温39.5℃，畏寒，热感。")])
        self.assertEqual(self.extract("精神欠佳，鼻导管吸氧，食欲略差。"), [])

    def test_pathogen_names_not_pneumonia_diagnosis(self):
        result = self.extract("肺炎支原体、肺炎链球菌、肺炎克雷伯菌。")
        self.assertEqual(len(result), 3)
        self.assertNotIn("肺炎诊断", [label for _, label in result])

    def test_longest_nested_pathogen_and_imaging_priority(self):
        self.assertEqual(self.extract("副流感病毒"), [("副流感病毒", "副流感病毒")])
        self.assertEqual(self.extract("CT提示右肺炎症。"), [("右肺炎症", "肺炎影像")])
        self.assertEqual(self.extract("诊断社区获得性肺炎。"), [("社区获得性肺炎", "肺炎诊断")])

    def test_context_exclusions_and_qualifiers(self):
        result = self.extract("吸气性喘鸣，休克指数升高，予脱水治疗。")
        self.assertEqual(result, [])
        self.assertIn(("感染性休克", "休克"), self.extract("感染性休克。"))
        self.assertIn(("急性Ⅰ型呼吸衰竭", "呼吸衰竭"), self.extract("急性Ⅰ型呼吸衰竭。"))
        self.assertIn(("接触病死禽畜", "可疑动物或动物制品接触史"), self.extract("接触病死禽畜。"))

    def test_capture_offsets_and_distinct_repeated_measurements(self):
        text = "🧪体温38.5℃，体温39.0℃。"
        entities = predict_entities(text, self.rules)
        numbers = [e for e in entities if e["label"] == "数值"]
        self.assertEqual([e["text"] for e in numbers], ["38.5", "39.0"])
        self.assertEqual(numbers[0]["start"], 3)
        self.assertTrue(all(text[e["start"]:e["end"]] == e["text"] for e in entities))

    def test_named_capture_excludes_context_and_duplicate_patterns_deduplicate(self):
        rules = compile_rules({"version": "test", "rules": [
            {"label": "发热", "patterns": ["提示(?P<entity>发热)", "发热"]},
        ]}, {"symptons_labels": ["发热"]})
        self.assertEqual(predict_entities("提示发热", rules), [
            {"start": 2, "end": 4, "label": "发热", "text": "发热"},
        ])

    def test_cross_control_overlap_is_allowed(self):
        rules = compile_rules({"version": "test", "rules": [
            {"label": "a", "patterns": ["abc"]}, {"label": "b", "patterns": ["bc"]},
        ]}, {"one": ["a"], "two": ["b"]})
        self.assertEqual(len(predict_entities("abc", rules)), 2)

    def test_invalid_rule_contracts_fail(self):
        bad = [
            {"version": "test", "rules": []},
            {"rules": self.payload["rules"]},
        ]
        for field, value in (("patterns", ["a*"]), ("priority", True), ("label", "unknown"), ("exclude_before", "")):
            payload = copy.deepcopy(self.payload)
            payload["rules"][0][field] = value
            bad.append(payload)
        payload = copy.deepcopy(self.payload)
        payload["rules"].append(payload["rules"][0])
        bad.append(payload)
        for payload in bad:
            with self.subTest(payload=payload.get("version")), self.assertRaises(ValueError):
                compile_rules(payload, self.groups)
        rules = compile_rules({"version": "test", "rules": [{"label": "a", "patterns": ["(?=a)"]}]}, {"one": ["a"]})
        with self.assertRaises(ValueError):
            predict_entities("a", rules)

    def test_inference_ignores_gold_and_timing_preserves_negative_records(self):
        tasks = [
            {"task_id": "positive", "text": "无发热。", "entities": "deliberately unreadable gold"},
            {"task_id": "negative", "text": "一般情况可。", "entities": [{"label": "made-up"}]},
        ]
        predictions, timing = run_baseline(tasks, self.groups, self.rules, repeats=3, warmup=2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[1]["entities"], [])
        self.assertEqual(timing["measured_task_runs"], 6)
        self.assertEqual(timing["successful_task_runs"], 6)
        self.assertEqual(timing["warmup_passes"], 2)
        self.assertEqual(len(timing["samples"]), 6)
        self.assertGreater(timing["successful_tasks_per_second"], 0)
        self.assertGreaterEqual(timing["p95_latency_ms"], timing["p50_latency_ms"])
        self.assertNotIn("一般情况可", json.dumps(timing, ensure_ascii=False))

    def test_invalid_tasks_and_timing_settings_fail(self):
        task = {"task_id": "1", "text": "发热"}
        for tasks, repeats, warmup in (([], 1, 0), ([task, task], 1, 0), ([task], 0, 0), ([task], 1, -1)):
            with self.subTest(tasks=tasks, repeats=repeats, warmup=warmup), self.assertRaises(ValueError):
                run_baseline(tasks, self.groups, self.rules, repeats, warmup)

    def test_cli_generates_predictions_timing_and_hashes_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks, predictions, benchmark = [root / name for name in ("tasks.json", "predictions.json", "benchmark.json")]
            tasks.write_text(json.dumps([{"task_id": "fictional", "text": "无发热。"}], ensure_ascii=False), encoding="utf-8")
            args = ["--tasks", str(tasks), "--rules", str(RULES), "--label-config", str(CONFIG),
                    "--output", str(predictions), "--benchmark", str(benchmark), "--dataset-kind", "synthetic_demo", "--repeats", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(benchmark.read_text(encoding="utf-8"))
            self.assertEqual(result["dataset_kind"], "synthetic_demo")
            self.assertEqual(result["measured_task_runs"], 2)
            self.assertEqual(len(result["inputs"]["rules"]["sha256"]), 64)
            contents = tasks.read_bytes()
            args[args.index("--output") + 1] = str(tasks)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(tasks.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
