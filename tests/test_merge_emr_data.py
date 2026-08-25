import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_emr_data import (
    build_label_studio_task,
    build_combined_text,
    calculate_age,
    classify_age_group,
    merge_data,
    split_age_groups,
    write_report,
    write_tasks_json,
)


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

    def test_writers_emit_label_studio_compatible_utf8_json(self):
        records = [
            {
                "patient_id": "患者一",
                "serial_number": "1",
                "text": "发热三天",
                "patient_info": [{"patient_name": "张三"}],
            }
        ]
        report = {"output_records": 1}
        json_path = self.root / "out" / "merged.json"
        report_path = self.root / "out" / "report.json"

        write_tasks_json(records, json_path)
        write_report(report, report_path)

        tasks = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIsInstance(tasks, list)
        self.assertEqual(1, len(tasks))
        task = tasks[0]
        self.assertEqual({"data"}, set(task))
        self.assertEqual("患者一", task["data"]["patient_id"])
        self.assertEqual("张三", task["data"]["patient_name"])
        self.assertEqual("发热三天", task["data"]["text"])
        self.assertEqual(records[0]["patient_info"], task["data"]["patient_info"])
        self.assertEqual(report, json.loads(report_path.read_text(encoding="utf-8")))

    def test_writer_emits_empty_json_list_for_no_records(self):
        json_path = self.root / "out" / "empty.json"

        write_tasks_json([], json_path)

        self.assertEqual([], json.loads(json_path.read_text(encoding="utf-8")))

    def test_label_studio_task_supplies_all_xml_variables(self):
        task = build_label_studio_task(
            {
                "patient_id": "p1",
                "serial_number": "s1",
                "text": "咳嗽伴发热",
                "patient_info": [
                    {"patient_name": ""},
                    {"patient_name": "患者甲"},
                ],
                "encounter_data": {"emr_order": [{"id": "o1"}]},
            }
        )

        self.assertEqual("p1", task["data"]["patient_id"])
        self.assertEqual("患者甲", task["data"]["patient_name"])
        self.assertEqual("咳嗽伴发热", task["data"]["text"])
        self.assertEqual(
            {"emr_order": [{"id": "o1"}]}, task["data"]["encounter_data"]
        )

    def test_label_studio_task_uses_empty_name_when_patient_info_is_missing(self):
        task = build_label_studio_task(
            {"patient_id": "p1", "serial_number": "s1", "text": "发热"}
        )

        self.assertEqual("", task["data"]["patient_name"])

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
                    {
                        "chief_complaint": "咳嗽三天",
                        "present_illness_his": "体温39℃",
                    }
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

        text, counts, duplicates = build_combined_text(record)

        expected_fragments = [
            "【主诉】\n咳嗽三天",
            "【体格检查】\n双肺呼吸音粗",
            "【现病史】\n体温39℃",
            "【检查备注】\n血常规",
            "【检验项目】\n白细胞计数",
            "【检验结果】\n升高",
            "【检验数值】\n12.3",
            "【检验单位】\n10^9/L",
        ]
        self.assertEqual("\n\n".join(expected_fragments), text)
        self.assertEqual(2, counts["emr_activity_info"])
        self.assertEqual(4, counts["emr_ex_lab_item"])
        self.assertEqual(1, duplicates["emr_outpatient_record"])

    def test_merge_adds_root_text_and_text_audit(self):
        main = self.write_csv(
            "temp_202608_emr_activity_info.csv",
            [
                {
                    "id": "a1",
                    "patient_id": "p1",
                    "serial_number": "s1",
                    "chief_complaint": "咳嗽",
                    "id_card": "310101199001150010",
                    "activity_time": "2025-01-14 08:00:00.000",
                }
            ],
        )

        records, report = merge_data(self.data_dir, main)

        self.assertIn("【主诉】\n咳嗽", records[0]["text"])
        self.assertEqual(34, records[0]["age"])
        self.assertFalse(records[0]["age_is_approximate"])
        self.assertEqual("成人组", records[0]["age_group"])
        self.assertEqual(1, report["combined_text"]["records_with_text"])
        self.assertEqual(len(records[0]["text"]), report["combined_text"]["total_characters"])

    def test_age_falls_back_to_visible_year_for_masked_id_card(self):
        record = {
            "activity_info": [
                {
                    "id_card": "3101061940****0014",
                    "activity_time": "2025-09-12 09:11:26.000",
                }
            ]
        }

        age, approximate = calculate_age(record)

        self.assertEqual(85, age)
        self.assertTrue(approximate)

    def test_age_group_boundary_and_split(self):
        self.assertEqual("儿童组", classify_age_group(17))
        self.assertEqual("成人组", classify_age_group(18))
        self.assertIsNone(classify_age_group(None))

        children, adults, ungrouped = split_age_groups(
            [
                {"id": "c", "age_group": "儿童组"},
                {"id": "a", "age_group": "成人组"},
                {"id": "u", "age_group": None},
            ]
        )
        self.assertEqual(["c"], [record["id"] for record in children])
        self.assertEqual(["a"], [record["id"] for record in adults])
        self.assertEqual(["u"], [record["id"] for record in ungrouped])


if __name__ == "__main__":
    unittest.main()
