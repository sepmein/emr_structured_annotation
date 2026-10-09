import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_entity_predictions import (
    build_report,
    main,
    normalize_records,
    read_label_groups,
    render_markdown,
)


GROUPS = {"symptons_labels": ["发热", "休克"], "diagnosis_labels": ["肺炎诊断"]}
LABELS = {label for group in GROUPS.values() for label in group}


def task(task_id, entities, *, text="发热，无肺炎", status="ok"):
    return {"task_id": task_id, "text": text, "entities": entities, "status": status}


def entity(start=0, end=2, label="发热"):
    return {"start": start, "end": end, "label": label}


def records(payload, *, reference=False):
    return normalize_records(payload, LABELS, reference=reference, source="test")


class EntityEvaluationTests(unittest.TestCase):
    def test_three_way_regex_comparisons_use_same_reference_and_all_metrics(self):
        gold = records([task("1", [entity()]), task("2", [])], reference=True)
        regex = records([task("1", [entity()]), task("2", [entity()])])
        before = records([task("1", [])])
        after = records([task("1", [entity()]), task("2", [])])
        report = build_report(gold, before, GROUPS, after, regex)
        self.assertEqual(list(report["runs"]), ["regex", "before", "after"])
        delta = report["comparisons"]["after_vs_regex"]["发热"]
        self.assertEqual(delta["precision_change"], 0.5)
        self.assertEqual(delta["recall_change"], 0.0)
        self.assertAlmostEqual(delta["f1_change"], 1 / 3)
        self.assertEqual(delta["mention_accuracy_change"], 0.5)
        self.assertIsNone(report["comparisons"]["after_vs_regex"]["休克"]["f1_change"])
        self.assertEqual(report["runs"]["before"]["coverage"]["missing_prediction_tasks"], 1)
        self.assertIn("after_vs_regex", render_markdown(report))

    def test_regex_only_supported_and_empty_comparison_rejected(self):
        gold = records([task("1", [entity()])], reference=True)
        report = build_report(gold, None, GROUPS, regex=gold)
        self.assertEqual(list(report["runs"]), ["regex"])
        self.assertEqual(report["comparisons"], {})
        self.assertEqual(report["runs"]["regex"]["overall"]["micro"]["f1"], 1)
        with self.assertRaises(ValueError):
            build_report(gold, None, GROUPS)

    def test_regex_only_cli_marks_synthetic_data_and_records_hash(self):
        config = Path(__file__).resolve().parents[1] / "label_studio" / "pneumonia_config.xml"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out, md = [root / name for name in ("fictional.json", "metrics.json", "metrics.md")]
            source.write_text(json.dumps([task("1", [entity()])]), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--reference", str(source), "--regex", str(source), "--label-config", str(config),
                      "--output", str(out), "--markdown", str(md), "--dataset-kind", "synthetic_demo"])
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_kind"], "synthetic_demo")
            self.assertIn("regex", report["inputs"])
            self.assertIn("虚构数据演示", md.read_text(encoding="utf-8"))

    def test_exact_matching_and_before_after_delta(self):
        gold = records([task("1", [entity()]), task("2", [])], reference=True)
        before = records([task("1", [entity(end=1)]), task("2", [entity()])])
        after = records([task("1", [entity()]), task("2", [])])
        report = build_report(gold, before, GROUPS, after)
        b = report["runs"]["before"]["per_label"]["发热"]
        a = report["runs"]["after"]["per_label"]["发热"]
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (0, 2, 1))
        self.assertEqual((a["tp"], a["fp"], a["fn"]), (1, 0, 0))
        self.assertEqual(report["f1_change"]["发热"], 1.0)
        # Wrong span can still be a correct label mention, a different metric.
        self.assertEqual(report["runs"]["before"]["label_mention_presence"]["发热"]["accuracy"], 0.5)
        self.assertIn("+100.0", render_markdown(report))

    def test_duplicate_predictions_count_as_false_positive(self):
        gold = records([task(1, [entity()])], reference=True)
        pred = records([task(1, [entity(), entity()])])
        run = build_report(gold, pred, GROUPS)["runs"]["before"]
        m = run["per_label"]["发热"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(run["coverage"]["duplicate_prediction_entities"], 1)

    def test_missing_failed_and_negative_tasks_remain_in_denominator(self):
        gold = records([
            task("positive_missing", [entity()]),
            task("negative_failed", []),
            task("negative_valid", []),
        ], reference=True)
        pred = records([
            task("negative_failed", [], status="failed"),
            task("negative_valid", []),
        ])
        run = build_report(gold, pred, GROUPS)["runs"]["before"]
        self.assertEqual(run["coverage"]["reference_tasks"], 3)
        self.assertEqual(run["coverage"]["successful_prediction_tasks"], 1)
        self.assertEqual(run["coverage"]["missing_task_ids"], ["positive_missing"])
        self.assertEqual(run["coverage"]["failed_task_ids"], ["negative_failed"])
        self.assertEqual(run["per_label"]["发热"]["fn"], 1)
        self.assertEqual(run["per_label"]["发热"]["f1"], 0)
        presence = run["label_mention_presence"]["发热"]
        self.assertEqual(presence["failed_negative"], 1)
        self.assertEqual(presence["accuracy"], 1 / 3)

    def test_no_reference_examples_have_no_recall_or_f1_but_keep_false_positives(self):
        gold = records([task(1, [entity()])], reference=True)
        pred = records([task(1, [entity(label="休克")])])
        run = build_report(gold, pred, GROUPS)["runs"]["before"]
        m = run["per_label"]["休克"]
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["precision"], 0)
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["f1"])
        self.assertEqual(run["overall"]["micro"]["fp"], 1)
        self.assertEqual(run["overall"]["supported_label_count"], 1)
        self.assertIsNone(run["per_label"]["肺炎诊断"]["precision"])

    def test_wrong_label_produces_false_positive_and_false_negative(self):
        gold = records([task(1, [entity()])], reference=True)
        pred = records([task(1, [entity(label="休克")])])
        run = build_report(gold, pred, GROUPS)["runs"]["before"]
        self.assertEqual(run["per_label"]["发热"]["fn"], 1)
        self.assertEqual(run["per_label"]["休克"]["fp"], 1)
        self.assertEqual(run["overall"]["micro"]["tp"], 0)

    def test_changed_original_text_and_extra_tasks_are_rejected(self):
        gold = records([task(1, [entity()])], reference=True)
        for pred in (
            records([task(1, [entity()], text="发热，无肺炎\n")]),
            records([task(1, []), task(2, [])]),
        ):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                build_report(gold, pred, GROUPS)

    def test_invalid_records_rejected(self):
        bad_inputs = [
            [task(1, [entity(end=99)])],
            [task(1, [entity(start=True)])],
            [task(1, [entity(label="未知标签")])],
            [task(1, [{**entity(), "text": "错误原文"}])],
            [task(1, [entity()], status="failed")],
            [task(1, []), task("1", [])],
            [{"task_id": 1, "text": "发热"}],
            [{"task_id": True, "text": "发热", "entities": []}],
            [task(1, [], status="not_finished")],
            [task(1, [], status=[])],
        ]
        for payload in bad_inputs:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                records(payload)
        for payload in ([], [task(1, [entity(), entity()])], [task(1, [], status="failed")]):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                records(payload, reference=True)

    def test_cli_reports_all_current_labels_and_does_not_copy_original_text(self):
        config = Path(__file__).resolve().parents[1] / "label_studio" / "pneumonia_config.xml"
        groups = read_label_groups(config)
        self.assertEqual(sum(map(len, groups.values())), 49)
        self.assertEqual(len(groups["symptons_labels"]), 12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path, before_path, after_path = [root / name for name in ("gold.json", "before.json", "after.json")]
            text = "虚构测试：发热"
            gold_path.write_text(json.dumps([task("fake1", [entity(5, 7)], text=text)], ensure_ascii=False), encoding="utf-8-sig")
            before_path.write_text(json.dumps([task("fake1", [], text=text)]), encoding="utf-8")
            after_path.write_text(json.dumps([task("fake1", [entity(5, 7)], text=text)]), encoding="utf-8")
            out, md = root / "results" / "metrics.json", root / "results" / "metrics.md"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([
                    "--reference", str(gold_path), "--before", str(before_path),
                    "--after", str(after_path), "--label-config", str(config),
                    "--output", str(out), "--markdown", str(md),
                ])
            self.assertEqual(result, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]["after"]["per_label"]), 49)
            self.assertEqual(report["f1_change"]["发热"], 1.0)
            self.assertEqual(len(report["inputs"]["reference"]["sha256"]), 64)
            for path in (out, md):
                self.assertNotIn(text, path.read_text(encoding="utf-8"))

    def test_cli_cannot_overwrite_reference(self):
        config = Path(__file__).resolve().parents[1] / "label_studio" / "pneumonia_config.xml"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.json"
            contents = json.dumps([task("1", [entity()])])
            path.write_text(contents, encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--reference", str(path), "--before", str(path), "--label-config", str(config), "--output", str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), contents)


if __name__ == "__main__":
    unittest.main()
