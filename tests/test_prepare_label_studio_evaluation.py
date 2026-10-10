import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from emr_annotation.annotation_analysis.label_studio_evaluation import audit_export, load_schema, main, select_reference
from emr_annotation.evaluation.entity_predictions import normalize_records, build_report


CONFIG = Path(__file__).resolve().parents[1] / "label_studio" / "pneumonia_config.global-single.xml"


def entity(region="entity1", label="发热", start=0, end=2, control="symptons_labels"):
    return {"id": region, "type": "labels", "from_name": control, "to_name": "chief_complaint_text", "value": {"start": start, "end": end, "labels": [label]}}


def choice(name="case_decision", value="待专业复核", region="case1"):
    return {"id": region, "type": "choices", "from_name": name, "to_name": "chief_complaint_text", "value": {"choices": [value]}}


def annotation(ann_id=10, annotator=1, results=None):
    return {"id": ann_id, "completed_by": annotator, "was_cancelled": False, "result": results if results is not None else [entity(), choice()]}


def task(task_id=1, annotations=None):
    return {"id": task_id, "data": {"text": "发热三天", "patient_id": "fictional-p1"}, "annotations": annotations if annotations is not None else [annotation()]}


class LabelStudioPreparationTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(CONFIG)

    def test_pair_agreement_ignores_region_ids_but_not_attribute_values(self):
        a = annotation(10, 1, [entity("a"), choice(), choice("finding_context", "明确存在", "a")])
        b = annotation(11, 2, [entity("b"), choice(), choice("finding_context", "明确存在", "b")])
        audit, _ = audit_export([task(annotations=[a, b])], self.schema)
        self.assertEqual(audit["summary"]["two_distinct_annotator_tasks"], 1)
        self.assertTrue(all(audit["tasks"][0]["pair_exact_matches"].values()))
        b["result"][-1]["value"]["choices"] = ["明确不存在"]
        audit, _ = audit_export([task(annotations=[a, b])], self.schema)
        self.assertEqual(audit["pair_exact_agreement"]["attributes"]["rate"], 0)
        self.assertEqual(audit["pair_exact_agreement"]["entities"]["rate"], 1)

    def test_two_annotations_by_same_or_unknown_person_are_not_independent_pairs(self):
        for second in (1, None):
            with self.subTest(second=second):
                audit, _ = audit_export([task(annotations=[annotation(10, 1), annotation(11, second)])], self.schema)
                self.assertFalse(audit["tasks"][0]["pair_eligible"])
                self.assertIsNone(audit["pair_exact_agreement"]["entities"]["rate"])

    def test_repeated_annotation_ids_invalidate_all_copies(self):
        audit, tasks = audit_export([task(annotations=[annotation(10, 1), annotation(10, 2)])], self.schema)
        self.assertEqual(tasks["1"]["annotations"], {})
        self.assertEqual(audit["summary"]["valid_annotations"], 0)
        self.assertEqual(audit["summary"]["invalid_annotations"], 2)
        self.assertTrue(all(a["status"] == "invalid" for a in audit["tasks"][0]["annotations"]))

    def test_cancelled_empty_invalid_and_predictions_do_not_become_reference_candidates(self):
        cancelled = annotation(12)
        cancelled["was_cancelled"] = True
        invalid = annotation(14, results=[entity(end=99), choice()])
        t = task(annotations=[annotation(10), cancelled, annotation(13, results=[]), invalid])
        t["predictions"] = [{"result": [entity(), choice()]}]
        audit, tasks = audit_export([t], self.schema)
        self.assertEqual(list(tasks["1"]["annotations"]), ["10"])
        self.assertEqual(audit["summary"]["valid_annotations"], 1)
        self.assertEqual(audit["summary"]["cancelled_annotations"], 1)
        self.assertEqual(audit["summary"]["invalid_annotations"], 2)

    def test_entities_attributes_relations_and_case_choice_survive_selection(self):
        time = entity("time", "时间表达", 2, 4)
        rel = {"type": "relation", "from_id": "symptom", "to_id": "time", "labels": ["持续时长"], "direction": "right"}
        a = annotation(results=[rel, choice("finding_context", "明确存在", "symptom"), choice(), time, entity("symptom")])
        _, tasks = audit_export([task(annotations=[a])], self.schema)
        provenance = {"source_sha256": "source", "label_config_sha256": "config"}
        selection = {**provenance, "selections": [{"task_id": "1", "action": "include", "annotation_id": 10, "reviewed": True, "reviewer": "reviewer-1"}]}
        reference, receipt = select_reference(tasks, selection, provenance)
        self.assertEqual(len(reference[0]["entities"]), 2)
        self.assertEqual(reference[0]["relations"][0]["type"], "持续时长")
        self.assertEqual(reference[0]["attributes"][0]["region_id"], "symptom")
        self.assertEqual(reference[0]["case_choices"]["case_decision"], "待专业复核")
        self.assertEqual(reference[0]["source_annotation_id"], "10")
        self.assertEqual(reference[0]["patient_id"], "fictional-p1")
        self.assertEqual(receipt["included"][0]["reviewer"], "reviewer-1")
        labels = {label for group in self.schema["groups"].values() for label in group}
        normalized = normalize_records(reference, labels, reference=True, source="selected")
        report = build_report(normalized, normalized, self.schema["groups"])
        self.assertEqual(report["runs"]["before"]["overall"]["micro"]["f1"], 1)

    def test_reviewed_case_only_negative_retained_but_empty_result_is_invalid(self):
        _, tasks = audit_export([task(annotations=[annotation(results=[choice(value="非目标")])])], self.schema)
        provenance = {"source_sha256": "s", "label_config_sha256": "c"}
        selection = {**provenance, "selections": [{"task_id": 1, "action": "include", "annotation_id": 10, "reviewed": True, "reviewer": "r"}]}
        reference, _ = select_reference(tasks, selection, provenance)
        self.assertEqual(reference[0]["entities"], [])
        self.assertEqual(reference[0]["case_choices"]["case_decision"], "非目标")

    def test_single_entity_rejects_multiple_labels_but_distinct_entities_keep_attributes(self):
        multi = entity(label="发热")
        multi["value"]["labels"] = ["发热", "肺炎诊断"]
        audit, _ = audit_export([task(annotations=[annotation(results=[multi, choice()])])], self.schema)
        self.assertEqual(audit["summary"]["invalid_annotations"], 1)
        results = [entity("fever"), entity("diagnosis", "肺炎诊断", 2, 4), choice(),
                   choice("finding_context", "明确存在", "fever"),
                   choice("temporality", "当前", "fever")]
        audit, tasks = audit_export([task(annotations=[annotation(results=results)])], self.schema)
        self.assertEqual(audit["summary"]["valid_annotations"], 1)
        parsed = tasks["1"]["annotations"]["10"]
        self.assertEqual(len(parsed["entities"]), 2)
        self.assertEqual(len(parsed["attributes"]), 2)

    def test_historical_control_names_require_archived_schema(self):
        archive = CONFIG.parent / "archive/pneumonia_config.before-global-single.xml"
        old = load_schema(archive)
        results = [entity("time", "时间表达", 2, 4, "time_labels"), entity("fever"), choice()]
        payload = [task(annotations=[annotation(results=results)])]
        self.assertEqual(audit_export(payload, old)[0]["summary"]["valid_annotations"], 1)
        self.assertEqual(audit_export(payload, self.schema)[0]["summary"]["invalid_annotations"], 1)

    def test_wrong_control_dangling_links_and_missing_required_case_rejected(self):
        bad_results = [
            [entity(control="diagnosis_labels"), choice()],
            [entity(), choice("finding_context", "明确存在", "missing"), choice()],
            [entity(), {"type": "relation", "from_id": "entity1", "to_id": "missing", "labels": ["持续时长"]}, choice()],
            [entity()],
            [entity(), choice(value="未知类别")],
            [entity(), choice(), choice()],
            [entity(), choice("finding_context", "明确存在", "entity1"), choice("finding_context", "明确存在", "entity1"), choice()],
            [{**entity(), "from_name": []}, choice()],
            [entity(), entity("time", "时间表达", 2, 4), {"type": "relation", "from_id": "entity1", "to_id": "time", "labels": ["持续时长"], "direction": []}, choice()],
        ]
        for results in bad_results:
            with self.subTest(results=results):
                audit, tasks = audit_export([task(annotations=[annotation(results=results)])], self.schema)
                self.assertFalse(tasks["1"]["annotations"])
                self.assertEqual(audit["summary"]["invalid_annotations"], 1)

    def test_selection_requires_review_current_hashes_and_complete_task_accounting(self):
        _, tasks = audit_export([task()], self.schema)
        provenance = {"source_sha256": "s", "label_config_sha256": "c"}
        row = {"task_id": 1, "action": "include", "annotation_id": 10, "reviewed": True, "reviewer": "r"}
        base = {**provenance, "selections": [row]}
        invalid = [
            {**base, "source_sha256": "stale"},
            {**base, "selections": []},
            {**base, "selections": [row, row]},
            {**base, "selections": [{**row, "reviewed": False}]},
            {**base, "selections": [{**row, "reviewer": ""}]},
            {**base, "selections": [{**row, "action": "pending"}]},
            {**base, "selections": [{**row, "annotation_id": 999}]},
            {**base, "selections": [{**row, "action": "exclude", "reason": ""}]},
        ]
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                select_reference(tasks, selection, provenance)

    def test_explicit_exclusion_is_recorded_and_missing_patient_id_is_counted(self):
        t = task(2)
        del t["data"]["patient_id"]
        audit, tasks = audit_export([task(), t], self.schema)
        provenance = {"source_sha256": "s", "label_config_sha256": "c"}
        selection = {**provenance, "selections": [
            {"task_id": 1, "action": "include", "annotation_id": 10, "reviewed": True, "reviewer": "r"},
            {"task_id": 2, "action": "exclude", "reviewed": True, "reviewer": "r", "reason": "source context incomplete"},
        ]}
        reference, receipt = select_reference(tasks, selection, provenance)
        self.assertEqual(audit["summary"]["tasks_missing_patient_id"], 1)
        self.assertEqual(len(reference), 1)
        self.assertEqual(receipt["excluded"][0]["task_id"], "2")

    def test_cli_never_auto_selects_and_generates_scorer_compatible_reference_after_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / "export.json"
            export.write_text(json.dumps([task(annotations=[annotation(10, 1), annotation(11, 2)])], ensure_ascii=False), encoding="utf-8-sig")
            inspect = root / "inspect"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--export", str(export), "--label-config", str(CONFIG), "--output-dir", str(inspect)]), 0)
            self.assertFalse((inspect / "reference.json").exists())
            template_path = inspect / "selection_template.json"
            selection = json.loads(template_path.read_text(encoding="utf-8"))
            self.assertIsNone(selection["selections"][0]["annotation_id"])
            selection["selections"][0].update({"action": "include", "annotation_id": "11", "reviewed": True, "reviewer": "r"})
            reviewed = root / "reviewed.json"
            reviewed.write_text(json.dumps(selection), encoding="utf-8")
            selected = root / "selected"
            args = ["--export", str(export), "--label-config", str(CONFIG), "--selection", str(reviewed), "--output-dir", str(selected)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            reference = json.loads((selected / "reference.json").read_text(encoding="utf-8"))
            self.assertEqual(reference[0]["source_annotation_id"], "11")
            self.assertEqual(export.read_text(encoding="utf-8-sig"), json.dumps([task(annotations=[annotation(10, 1), annotation(11, 2)])], ensure_ascii=False))
            original = (selected / "reference.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual((selected / "reference.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
