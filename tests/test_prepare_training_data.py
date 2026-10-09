import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_training_data import build_outputs, convert_offset_units, json_bytes, main
from scripts.prepare_label_studio_evaluation import audit_export, load_schema
from scripts.frozen_training_data import prepare_frozen_training


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "label_studio/pneumonia_config.xml"
EXPORT = ROOT / "tests/fixtures/label_studio_demo_export.json"
SELECTION = ROOT / "tests/fixtures/label_studio_demo_selection.json"


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.raw = EXPORT.read_bytes()
        self.selection = json.loads(SELECTION.read_text(encoding="utf-8"))

    def build(self, **kwargs):
        return build_outputs(self.raw, CONFIG, selection=self.selection, dataset_kind="synthetic_demo", **kwargs)

    def test_full_export_round_trip_keeps_one_reviewed_annotation_and_negatives(self):
        outputs = self.build(ratios=(0.6, 0.2, 0.2))
        audit, report = outputs["audit.json"], outputs["conversion_report.json"]
        self.assertEqual(audit["summary"]["valid_annotations"], 40)
        self.assertEqual(report["included_tasks"], 20)
        self.assertEqual(report["entities"], 28)
        self.assertEqual(report["no_entity_tasks"], 2)
        self.assertEqual(report["relations_in_model_data"], 5)
        self.assertEqual(report["split_tasks"], {"train": 12, "validation": 4, "test": 4})
        reference = outputs["reference.json"]
        # The first annotation intentionally misses fever; explicit review chooses the second.
        self.assertEqual(reference[0]["source_annotation_id"], "1012")
        self.assertEqual(reference[0]["entities"][0]["label"], "发热")
        # A false preannotation on task 14 never becomes reference/training data.
        self.assertEqual(reference[13]["entities"], [])
        self.assertEqual(outputs["training_data.json"][13]["entities"], [])
        for row in reference:
            for entity in row["entities"]:
                self.assertEqual(row["text"][entity["start"]:entity["end"]], entity["text"])
        for row in outputs["training_data.json"]:
            ids = {e["id"] for e in row["entities"]}
            for relation in row["relations"]:
                self.assertIn(relation["from_id"], ids)
                self.assertIn(relation["to_id"], ids)

    def test_export_aliases_normalize_and_original_values_survive(self):
        outputs = self.build()
        row = outputs["reference.json"][0]
        finding = next(a for a in row["attributes"] if a["name"] == "finding_context")
        self.assertEqual(finding["values"], ["明确不存在"])
        self.assertEqual(finding["export_values"], ["known_absent"])
        self.assertEqual(row["case_choices"]["case_decision"], "待专业复核")
        self.assertNotIn("attributes", outputs["training_data.json"][0])
        self.assertNotIn("case_choices", outputs["training_data.json"][0])
        self.assertNotIn("splits/train.json", outputs)

    def test_alias_and_display_value_compare_equally(self):
        payload = json.loads(self.raw)
        task = payload[2]
        schema = load_schema(CONFIG)
        for result in task["annotations"][0]["result"]:
            if result["type"] == "choices":
                control = schema["choices"][result["from_name"]]
                result["value"]["choices"] = [control["value_map"][v] for v in result["value"]["choices"]]
        audit, _ = audit_export([task], schema)
        self.assertTrue(all(audit["tasks"][0]["pair_exact_matches"].values()))
        for result in task["annotations"][0]["result"]:
            if result.get("from_name") == "finding_context":
                result["value"]["choices"] = ["made_up_alias"]
                break
        audit, _ = audit_export([task], schema)
        self.assertEqual(audit["summary"]["invalid_annotations"], 1)

    def test_alias_collisions_in_config_rejected(self):
        text = CONFIG.read_text(encoding="utf-8").replace('alias="known_absent"', 'alias="known_present"')
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.xml"
            config.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                load_schema(config)

    def test_audit_only_requires_review_before_training(self):
        outputs = build_outputs(self.raw, CONFIG)
        self.assertNotIn("reference.json", outputs)
        self.assertNotIn("training_data.json", outputs)
        template = outputs["selection_template.json"]
        self.assertEqual(template["offset_unit_in"], "codepoint")
        self.assertTrue(all(not s["reviewed"] and s["action"] == "pending" for s in template["selections"]))
        with self.assertRaises(ValueError):
            build_outputs(self.raw, CONFIG, ratios=(0.6, 0.2, 0.2))
        with self.assertRaises(ValueError):
            build_outputs(self.raw, CONFIG, selection=template)

    def test_modified_export_or_config_hash_rejected(self):
        with self.assertRaises(ValueError):
            build_outputs(self.raw + b"\n", CONFIG, selection=self.selection)
        selection = copy.deepcopy(self.selection)
        selection["label_config_sha256"] = "stale"
        with self.assertRaises(ValueError):
            build_outputs(self.raw, CONFIG, selection=selection)

    def test_frozen_splits_feed_training_preflight_without_leakage(self):
        outputs = self.build(ratios=(0.6, 0.2, 0.2))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in outputs.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(json_bytes(value))
            train, validation, receipt = prepare_frozen_training(directory / "splits", directory / "reference.json", CONFIG)
        self.assertEqual(train, outputs["model_data/train.json"])
        self.assertEqual(validation, outputs["model_data/validation.json"])
        self.assertTrue(receipt["verified_no_patient_overlap"])
        self.assertFalse(receipt["training_policy"]["test_passed_to_training"])
        train_ids = {r["task_id"] for r in train}
        self.assertFalse(train_ids & {r["task_id"] for r in outputs["model_data/test.json"]})

    def emoji_export(self):
        task = copy.deepcopy(json.loads(self.raw)[0])
        task["data"]["text"] = "🧪发热"
        task["annotations"] = [task["annotations"][1]]
        for result in task["annotations"][0]["result"]:
            if result["type"] == "labels":
                result["value"].update(start=2, end=4, text="发热")
        return [task]

    def test_utf16_conversion_is_explicit_and_never_changes_input_text(self):
        raw = self.emoji_export()
        before = copy.deepcopy(raw)
        normalized = convert_offset_units(raw, "utf16")
        self.assertEqual(raw, before)
        self.assertEqual(normalized[0]["data"]["text"], "🧪发热")
        entity = next(r for r in normalized[0]["annotations"][0]["result"] if r["type"] == "labels")
        self.assertEqual((entity["value"]["start"], entity["value"]["end"]), (1, 3))
        self.assertEqual(audit_export(normalized, load_schema(CONFIG))[0]["summary"]["valid_annotations"], 1)
        self.assertEqual(audit_export(raw, load_schema(CONFIG))[0]["summary"]["invalid_annotations"], 1)

    def test_surrogate_midpoint_is_invalid_instead_of_repaired(self):
        raw = self.emoji_export()
        next(r for r in raw[0]["annotations"][0]["result"] if r["type"] == "labels")["value"]["start"] = 1
        normalized = convert_offset_units(raw, "utf16")
        self.assertEqual(audit_export(normalized, load_schema(CONFIG))[0]["summary"]["invalid_annotations"], 1)

    def test_review_selection_bound_to_offset_unit(self):
        with self.assertRaisesRegex(ValueError, "offset_unit_in"):
            self.build(offset_unit="utf16")
        raw = json_bytes(self.emoji_export())
        selection = copy.deepcopy(self.selection)
        selection.update(source_sha256=hashlib.sha256(raw).hexdigest(), offset_unit_in="utf16")
        selection["selections"] = selection["selections"][:1]
        outputs = build_outputs(raw, CONFIG, selection=selection, offset_unit="utf16")
        self.assertEqual(outputs["training_data.json"][0]["entities"][0]["start"], 1)

    def test_cli_publishes_complete_result_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "converted"
            args = ["--export", str(EXPORT), "--label-config", str(CONFIG), "--selection", str(SELECTION),
                    "--ratios", "0.6", "0.2", "0.2", "--dataset-kind", "synthetic_demo", "--output-dir", str(target)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertTrue((target / "frozen_training_preflight.json").exists())
            before = (target / "conversion_report.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual((target / "conversion_report.json").read_bytes(), before)
            bad = Path(temp) / "invalid"
            args[-1] = str(bad)
            args[args.index("0.6")] = "0.9"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(bad.exists())


if __name__ == "__main__":
    unittest.main()
