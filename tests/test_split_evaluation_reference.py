import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.split_evaluation_reference import main, split_reference


GROUPS = {"symptons_labels": ["发热", "休克", "惊厥"]}
CONFIG = Path(__file__).resolve().parents[1] / "label_studio" / "pneumonia_config.xml"


def row(task_id, patient, text, label=None):
    return {
        "task_id": task_id, "patient_id": patient, "text": text,
        "entities": [{"start": 0, "end": 2, "label": label}] if label else [],
        "case_choices": {"case_decision": "待专业复核" if label else "非目标"},
        "source_annotation_id": f"reviewed-{task_id}",
    }


def sample():
    return [
        row("1", "p1", "发热", "发热"),
        row("2", "p1", "发热一天", "发热"),
        row("3", "p2", "发热一天", "发热"),
        row("4", "p3", "没有症状"),
        row("5", "p4", "普通复诊"),
        row("6", "p5", "休克", "休克"),
    ]


class ReferenceSplitTests(unittest.TestCase):
    def test_connected_patient_and_text_chains_cannot_cross_splits(self):
        splits, manifest = split_reference(sample(), GROUPS, (0.6, 0.2, 0.2), 42)
        assignment = {r["task_id"]: name for name, rows in splits.items() for r in rows}
        self.assertEqual(assignment["1"], assignment["2"])
        self.assertEqual(assignment["2"], assignment["3"])
        for name, rows in splits.items():
            self.assertTrue(rows)
            for other, other_rows in splits.items():
                if other != name:
                    self.assertFalse({r["patient_id"] for r in rows} & {r["patient_id"] for r in other_rows})
                    self.assertFalse({r["text"] for r in rows} & {r["text"] for r in other_rows})
        self.assertEqual(manifest["independent_groups"], 4)
        self.assertEqual(sum(len(rows) for rows in splits.values()), 6)
        self.assertTrue(manifest["verified_no_patient_overlap"])

    def test_seeded_split_is_reproducible_and_input_order_independent(self):
        payload = sample()
        original = deepcopy(payload)
        a = split_reference(payload, GROUPS, (0.6, 0.2, 0.2), 73)
        b = split_reference(list(reversed(payload)), GROUPS, (0.6, 0.2, 0.2), 73)
        self.assertEqual(a, b)
        self.assertEqual(payload, original)

    def test_negative_tasks_and_all_metadata_preserved(self):
        payload = sample()
        splits, manifest = split_reference(payload, GROUPS, (0.6, 0.2, 0.2), 42)
        flattened = {r["task_id"]: r for rows in splits.values() for r in rows}
        self.assertEqual(flattened, {r["task_id"]: r for r in payload})
        self.assertEqual(sum(s["no_entity_tasks"] for s in manifest["splits"].values()), 2)
        self.assertEqual(sum(s["case_decision_counts"].get("非目标", 0) for s in manifest["splits"].values()), 2)

    def test_label_coverage_is_reported_without_fabricating_missing_classes(self):
        _, manifest = split_reference(sample(), GROUPS, (0.6, 0.2, 0.2), 42)
        self.assertEqual(manifest["labels_absent_from_entire_batch"], ["惊厥"])
        self.assertEqual(sum(manifest["label_coverage"]["发热"]["entities"].values()), 3)
        self.assertEqual(sum(manifest["label_coverage"]["休克"]["positive_tasks"].values()), 1)
        self.assertTrue(manifest["coverage_warnings"])
        self.assertEqual(manifest["near_duplicate_check"], "not_performed")

    def test_missing_patient_id_never_falls_back_to_task_id(self):
        for patient in (None, "", "  ", False):
            payload = sample()
            payload[0]["patient_id"] = patient
            with self.subTest(patient=patient), self.assertRaises(ValueError):
                split_reference(payload, GROUPS, (0.6, 0.2, 0.2), 42)

    def test_fewer_than_three_independent_groups_rejected(self):
        payload = [row(str(i), "one-patient", f"发热{i}", "发热") for i in range(5)]
        with self.assertRaisesRegex(ValueError, "Fewer than three"):
            split_reference(payload, GROUPS, (0.6, 0.2, 0.2), 42)

    def test_invalid_ratios_and_invalid_reference_rejected(self):
        for ratios in ((0.8, 0.2, 0), (0.6, 0.2, 0.1), (float("nan"), 0.2, 0.2), (float("inf"), 0.2, 0.2), (-0.1, 0.5, 0.6)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_reference(sample(), GROUPS, ratios, 42)
        payload = sample()
        payload[0]["entities"][0]["end"] = 99
        with self.assertRaises(ValueError):
            split_reference(payload, GROUPS, (0.6, 0.2, 0.2), 42)

    def test_cli_persists_manifest_and_refuses_overwriting_frozen_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, out = root / "reference.json", root / "splits"
            reference.write_text(json.dumps(sample(), ensure_ascii=False), encoding="utf-8-sig")
            args = ["--reference", str(reference), "--label-config", str(CONFIG), "--ratios", "0.6", "0.2", "0.2", "--seed", "42", "--output-dir", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["label_coverage"]), 49)
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertEqual(manifest["tasks"], 6)
            files = {p.name: p.read_bytes() for p in out.iterdir()}
            self.assertEqual(set(files), {"train.json", "validation.json", "test.json", "split_manifest.json"})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(files, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
