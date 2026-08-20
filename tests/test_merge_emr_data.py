import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_emr_data import build_combined_text, merge_data, write_jsonl, write_report


class MergeEmrDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, rows):
        path = self.data_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_nested_merge_preserves_one_to_many_rows_without_cartesian_product(self):
        main = self.write_csv(
            "temp_202608_emr_activity_info.csv",
            [
                {"id": "a1", "patient_id": "p1", "serial_number": "s1", "disease": "d1"},
                {"id": "a2", "patient_id": "p1", "serial_number": "s1", "disease": "d2"},
                {"id": "a3", "patient_id": "p2", "serial_number": "s2", "disease": "d3"},
            ],
        )
        self.write_csv(
            "temp_202608_emr_patient_info.csv",
            [
                {"id": "p1", "name": "first"},
                {"id": "p1", "name": "second"},
                {"id": "p9", "name": "unmatched"},
            ],
        )
        self.write_csv(
            "temp_202608_emr_order.csv",
            [
                {"id": "o1", "patient_id": "p1", "serial_number": "s1", "date": "1"},
                {"id": "o2", "patient_id": "p1", "serial_number": "s1", "date": "2"},
                {"id": "o9", "patient_id": "p9", "serial_number": "s9", "date": "9"},
            ],
        )
        self.write_csv(
            "temp_202608_emr_order_item.csv",
            [
                {"id": "i1", "order_id": "o1", "drug": "A"},
                {"id": "i2", "order_id": "o1", "drug": "B"},
                {"id": "i9", "order_id": "missing", "drug": "X"},
            ],
        )

        records, report = merge_data(self.data_dir, main)

        self.assertEqual(2, len(records))
        first = next(record for record in records if record["patient_id"] == "p1")
        self.assertEqual(2, len(first["activity_info"]))
        self.assertEqual(2, len(first["patient_info"]))
        self.assertEqual(2, len(first["encounter_data"]["emr_order"]))
        self.assertEqual(2, len(first["encounter_data"]["emr_order"][0]["items"]))
        self.assertEqual(0, len(first["encounter_data"]["emr_order"][1]["items"]))
        self.assertEqual(1, len(first["encounter_data"]["emr_order"][0]["records"]))
        self.assertEqual(1, report["activity"]["duplicate_key_rows"])
        self.assertEqual(1, report["encounter_tables"]["emr_order"]["unmatched_rows"])
        self.assertEqual(1, report["child_tables"]["emr_order_item"]["unmatched_rows_to_parent"])

    def test_ambiguous_parent_id_does_not_duplicate_child_across_encounters(self):
        main = self.write_csv(
            "temp_202608_emr_activity_info.csv",
            [
                {"id": "a1", "patient_id": "p1", "serial_number": "s1"},
                {"id": "a2", "patient_id": "p2", "serial_number": "s2"},
            ],
        )
        self.write_csv(
            "temp_202608_emr_order.csv",
            [
                {"id": "shared", "patient_id": "p1", "serial_number": "s1"},
                {"id": "shared", "patient_id": "p2", "serial_number": "s2"},
            ],
        )
        self.write_csv(
            "temp_202608_emr_order_item.csv",
            [{"id": "i1", "order_id": "shared", "drug": "A"}],
        )

        records, report = merge_data(self.data_dir, main)

        entities = [record["encounter_data"]["emr_order"][0] for record in records]
        self.assertEqual([0, 0], [len(entity["items"]) for entity in entities])
        self.assertTrue(all(entity["items_status"] == "ambiguous_parent_id" for entity in entities))
        self.assertEqual(
            1,
            report["child_tables"]["emr_order_item"][
                "ambiguous_rows_to_multiple_encounters"
            ],
        )

    def test_writers_emit_valid_utf8_json(self):
        records = [{"patient_id": "患者一", "serial_number": "1"}]
        report = {"output_records": 1}
        jsonl_path = self.root / "out" / "merged.jsonl"
        report_path = self.root / "out" / "report.json"

        write_jsonl(records, jsonl_path)
        write_report(report, report_path)

        self.assertEqual(records[0], json.loads(jsonl_path.read_text(encoding="utf-8")))
        self.assertEqual(report, json.loads(report_path.read_text(encoding="utf-8")))

    def test_combined_text_uses_configured_order_and_skips_empty_values(self):
        record = {
            "activity_info": [
                {
                    "chief_complaint": "咳嗽三天",
                    "present_illness_his": "",
                    "physical_examination": "双肺呼吸音粗",
                }
            ],
            "encounter_data": {
                "emr_outpatient_record": [
                    {"chief_complaint": "发热", "present_illness_his": "体温39℃"}
                ],
                "emr_ex_lab": [
                    {
                        "records": [{"examination_notes": "血常规"}],
                        "items": [
                            {
                                "item_name": "白细胞计数",
                                "examination_result_name": "升高",
                                "examination_quantification": "12.3",
                                "examination_quantification_unit": "10^9/L",
                            }
                        ],
                    }
                ],
            },
        }

        text, counts = build_combined_text(record)

        expected_fragments = [
            "[emr_activity_info[1].chief_complaint]\n咳嗽三天",
            "[emr_activity_info[1].physical_examination]\n双肺呼吸音粗",
            "[emr_outpatient_record[1].chief_complaint]\n发热",
            "[emr_outpatient_record[1].present_illness_his]\n体温39℃",
            "[emr_ex_lab[1].examination_notes]\n血常规",
            "[emr_ex_lab_item[1].item_name]\n白细胞计数",
            "[emr_ex_lab_item[1].examination_result_name]\n升高",
            "[emr_ex_lab_item[1].examination_quantification]\n12.3",
            "[emr_ex_lab_item[1].examination_quantification_unit]\n10^9/L",
        ]
        self.assertEqual("\n\n".join(expected_fragments), text)
        self.assertEqual(2, counts["emr_activity_info"])
        self.assertEqual(4, counts["emr_ex_lab_item"])

    def test_merge_adds_root_text_and_text_audit(self):
        main = self.write_csv(
            "temp_202608_emr_activity_info.csv",
            [
                {
                    "id": "a1",
                    "patient_id": "p1",
                    "serial_number": "s1",
                    "chief_complaint": "咳嗽",
                }
            ],
        )

        records, report = merge_data(self.data_dir, main)

        self.assertIn("[emr_activity_info[1].chief_complaint]\n咳嗽", records[0]["text"])
        self.assertEqual(1, report["combined_text"]["records_with_text"])
        self.assertEqual(len(records[0]["text"]), report["combined_text"]["total_characters"])


if __name__ == "__main__":
    unittest.main()
