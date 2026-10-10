import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from emr_annotation.annotation_analysis.label_studio_evaluation import load_schema, parse_annotation


CONFIG = Path(__file__).resolve().parents[1] / "label_studio/pneumonia_config.xml"
ARCHIVE = CONFIG.parent / "archive/pneumonia_config.before-global-single.xml"


class LabelStudioConfigCompatibilityTests(unittest.TestCase):
    def test_existing_controls_labels_attributes_and_relations_remain_compatible(self):
        current, old = load_schema(CONFIG), load_schema(ARCHIVE)
        for field in ("groups", "text_control", "text_field", "relations"):
            self.assertEqual(current[field], old[field], field)
        self.assertEqual(set(current["choices"]), set(old["choices"]))
        for name, choice in old["choices"].items():
            preserved = current["choices"][name]
            self.assertTrue(choice["values"].issubset(preserved["values"]), name)
            for exported, display in choice["value_map"].items():
                self.assertEqual(preserved["value_map"][exported], display, name)
            for field in ("per_region", "required"):
                self.assertEqual(preserved[field], choice[field], name)
        self.assertEqual(len(current["groups"]), 9)
        self.assertEqual(set(current["entity_choices"].values()), {"single"})
        root = ET.parse(CONFIG).getroot()
        for control in root.iter("Labels"):
            self.assertEqual(control.get("toName"), "chief_complaint_text")
            self.assertEqual(control.get("choice"), "single")

    def test_reported_eleven_legacy_results_survive_without_rewriting(self):
        # Reproduce the five from_name values and counts in the save error.
        spans = [
            ("diagnosis_labels", "肺炎诊断"),
            ("epidemics_labels", "境内外旅居史"),
            ("measure_labels", "数值"), ("measure_labels", "单位"),
            ("measure_labels", "比较符"), ("measure_labels", "数值"),
            ("pathogen", "新冠病毒"), ("pathogen", "流感病毒"),
            ("measure_entities", "体温"), ("measure_entities", "呼吸频率"),
            ("measure_entities", "静息状态指氧饱和度"),
        ]
        text = "，".join(label for _, label in spans)
        results, offset = [], 0
        for index, (control, label) in enumerate(spans):
            results.append({
                "id": f"legacy-{index}", "type": "labels", "from_name": control,
                "to_name": "chief_complaint_text",
                "value": {"start": offset, "end": offset + len(label), "text": label,
                          "labels": [label]},
            })
            offset += len(label) + 1
        results.extend([
            {"id": "legacy-6", "type": "choices", "from_name": "test_result",
             "to_name": "chief_complaint_text", "value": {"choices": ["positive"]}},
            {"type": "relation", "from_id": "legacy-8", "to_id": "legacy-2",
             "labels": ["测量"], "direction": "right"},
            {"id": "case", "type": "choices", "from_name": "case_decision",
             "to_name": "chief_complaint_text", "value": {"choices": ["待专业复核"]}},
        ])
        annotation = {"result": results}
        original = copy.deepcopy(annotation)
        parsed = parse_annotation(annotation, text, load_schema(CONFIG))
        self.assertEqual(annotation, original)
        self.assertEqual(len(parsed["entities"]), 11)
        self.assertEqual([entity["id"] for entity in parsed["entities"]],
                         [f"legacy-{index}" for index in range(11)])
        self.assertEqual(parsed["attributes"][0]["region_id"], "legacy-6")
        self.assertEqual(parsed["relations"][0]["from_id"], "legacy-8")
        with self.assertRaisesRegex(ValueError, "Unknown entity control"):
            parse_annotation(annotation, text,
                             load_schema(CONFIG.with_name("pneumonia_config.global-single.xml")))


if __name__ == "__main__":
    unittest.main()
