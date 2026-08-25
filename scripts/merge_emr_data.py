"""Merge EMR CSV exports into a Label Studio task JSON array.

The activity table is the driving table.  One output record is created for
each distinct (patient_id, serial_number) pair.  One-to-many relationships are
represented as arrays so that joining several detail tables cannot create a
Cartesian product.  The output is a JSON list of Label Studio tasks whose
encounter records are stored under the required ``data`` key.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


PATIENT_SUFFIX = "emr_patient_info.csv"
CHILD_RELATIONSHIPS = {
    "emr_ex_clinical.csv": ("emr_ex_clinical_item.csv", "ex_clinical_id"),
    "emr_ex_lab.csv": ("emr_ex_lab_item.csv", "ex_lab_id"),
    "emr_order.csv": ("emr_order_item.csv", "order_id"),
}

# Ordered according to the text-field relationship diagram.  The keys use the
# real CSV table/column names, including historical spelling in the exports.
TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "emr_activity_info": (
        "chief_complaint",
        "present_illness_his",
        "physical_examination",
        "studies_summary_result",
    ),
    "emr_outpatient_record": (
        "chief_complaint",
        "present_illness_his",
        "past_illness_his",
        "infection_his",
        "physical_examination",
        "personal_his",
        "studies_summary_result",
    ),
    "emr_outpatient_obs": (
        "chief_complaint",
        "present_illness_his",
        "personal_his",
        "studies_summary_result",
        "course",
    ),
    "emr_adminssion_info": (
        "chief_complaint",
        "present_illness_his",
        "past_illness_his",
        "infection_his",
        "physical_examination",
        "personal_his",
        "studies_summary_result",
        "specialized_examination",
    ),
    "emr_first_course": (
        "chief_complaint",
        "present_illness_his",
        "diagnosis_basis",
        "treatment_plan",
    ),
    "emr_daily_coure": ("course", "order_content", "treatment"),
    "emr_discharge_info": (
        "admission_desc",
        "studies_summary_result",
        "treatment_desc",
        "discharge_desc",
        "discharge_symptoms_signs",
    ),
    "emr_ex_clinical": (
        "symptom_desc",
        "treatment_desc",
        "examination_objective_desc",
        "examination_subjective_desc",
        "examination_notes",
    ),
    "emr_ex_lab": (
        "symptom_desc",
        "treatment_desc",
        "examination_objective_desc",
        "examination_subjective_desc",
        "examination_notes",
    ),
    "emr_ex_lab_item": (
        "item_name",
        "examination_result_name",
        "examination_quantification",
        "examination_quantification_unit",
    ),
}

TEXT_LABELS = {
    "chief_complaint": "主诉",
    "present_illness_his": "现病史",
    "physical_examination": "体格检查",
    "studies_summary_result": "检查摘要",
    "past_illness_his": "既往史",
    "infection_his": "传染病史",
    "personal_his": "个人史",
    "course": "病程记录",
    "specialized_examination": "专科检查",
    "diagnosis_basis": "诊断依据",
    "treatment_plan": "治疗计划",
    "order_content": "医嘱内容",
    "treatment": "诊疗措施",
    "admission_desc": "入院情况",
    "treatment_desc": "治疗经过",
    "discharge_desc": "出院情况",
    "discharge_symptoms_signs": "出院症状体征",
    "symptom_desc": "症状描述",
    "examination_objective_desc": "检查所见",
    "examination_subjective_desc": "检查结论",
    "examination_notes": "检查备注",
    "item_name": "检验项目",
    "examination_result_name": "检验结果",
    "examination_quantification": "检验数值",
    "examination_quantification_unit": "检验单位",
}

CHILD_GROUP = "儿童组"
ADULT_GROUP = "成人组"


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def encounter_key(row: dict[str, str]) -> tuple[str, str] | None:
    patient_id = _clean(row.get("patient_id"))
    serial_number = _clean(row.get("serial_number"))
    if not patient_id or not serial_number:
        return None
    return patient_id, serial_number


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while preserving identifiers as strings."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def logical_name(path: Path) -> str:
    """Return a batch-independent table name such as ``emr_ex_lab``."""
    return re.sub(r"^temp_\d+_", "", path.stem)


def _find_by_suffix(files: Iterable[Path], suffix: str) -> Path | None:
    matches = [path for path in files if path.name.endswith(suffix)]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple files match suffix {suffix!r}: {names}")
    return matches[0] if matches else None


def _index_rows(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    index: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(_clean(row.get(field)) for field in fields)
        if all(key):
            index[key].append(row)
    return dict(index)


def _table_audit(
    rows: list[dict[str, str]], main_keys: set[tuple[str, str]]
) -> dict[str, int | float]:
    keyed = [(row, encounter_key(row)) for row in rows]
    keyed_count = sum(key is not None for _, key in keyed)
    matched = sum(key in main_keys for _, key in keyed if key is not None)
    return {
        "rows": len(rows),
        "rows_with_key": keyed_count,
        "matched_rows": matched,
        "unmatched_rows": len(rows) - matched,
        "match_rate": round(matched / keyed_count, 6) if keyed_count else 0.0,
    }


def _text_rows(record: dict[str, Any], table: str) -> list[dict[str, Any]]:
    """Return raw rows for a configured text source in deterministic order."""
    if table == "emr_activity_info":
        return record.get("activity_info", [])

    encounter_data = record.get("encounter_data", {})
    if table == "emr_ex_lab_item":
        return [
            item
            for parent in encounter_data.get("emr_ex_lab", [])
            for item in parent.get("items", [])
        ]

    rows = encounter_data.get(table, [])
    if table in {"emr_ex_clinical", "emr_ex_lab", "emr_order"}:
        return [raw for parent in rows for raw in parent.get("records", [])]
    return rows


def build_combined_text(
    record: dict[str, Any],
) -> tuple[str, dict[str, int], dict[str, int]]:
    """Build Chinese-labeled encounter text, dropping exact duplicate values."""
    segments: list[str] = []
    source_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    seen_values: set[str] = set()
    for table, fields in TEXT_FIELDS.items():
        source_count = 0
        duplicate_count = 0
        for row in _text_rows(record, table):
            for field in fields:
                value = _clean(row.get(field))
                if not value:
                    continue
                if value in seen_values:
                    duplicate_count += 1
                    continue
                seen_values.add(value)
                segments.append(f"【{TEXT_LABELS[field]}】\n{value}")
                source_count += 1
        source_counts[table] = source_count
        duplicate_counts[table] = duplicate_count
    return "\n\n".join(segments), source_counts, duplicate_counts


def _parse_reference_date(record: dict[str, Any]) -> date | None:
    dates: list[date] = []
    for row in record.get("activity_info", []):
        value = _clean(row.get("activity_time"))
        if not value:
            continue
        try:
            dates.append(datetime.strptime(value[:10], "%Y-%m-%d").date())
        except ValueError:
            continue
    return min(dates) if dates else None


def _find_id_card(record: dict[str, Any]) -> str:
    for collection in (record.get("activity_info", []), record.get("patient_info", [])):
        for row in collection:
            value = _clean(row.get("id_card"))
            if value:
                return value.upper()
    return ""


def calculate_age(record: dict[str, Any]) -> tuple[int | None, bool | None]:
    """Calculate encounter-time age from a Chinese ID card.

    Fully specified 18/15-digit cards produce exact age.  A masked card with
    only a visible birth year produces a year-level estimate.
    """
    reference_date = _parse_reference_date(record)
    id_card = _find_id_card(record)
    if reference_date is None or not id_card:
        return None, None

    birth_date: date | None = None
    if re.fullmatch(r"\d{17}[0-9X]", id_card):
        token = id_card[6:14]
        try:
            birth_date = datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            birth_date = None
    elif re.fullmatch(r"\d{15}", id_card):
        token = "19" + id_card[6:12]
        try:
            birth_date = datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            birth_date = None

    if birth_date is not None:
        age = reference_date.year - birth_date.year - (
            (reference_date.month, reference_date.day)
            < (birth_date.month, birth_date.day)
        )
        return (age, False) if 0 <= age <= 150 else (None, None)

    year_match = re.match(r"^\d{6}((?:18|19|20)\d{2})", id_card)
    if year_match:
        age = reference_date.year - int(year_match.group(1))
        return (age, True) if 0 <= age <= 150 else (None, None)
    return None, None


def classify_age_group(age: int | None) -> str | None:
    """Classify a calculated age using the under-18 boundary."""
    if age is None:
        return None
    return CHILD_GROUP if age < 18 else ADULT_GROUP


def split_age_groups(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return child, adult, and ungrouped encounter records."""
    children: list[dict[str, Any]] = []
    adults: list[dict[str, Any]] = []
    ungrouped: list[dict[str, Any]] = []
    for record in records:
        group = record.get("age_group")
        if group == CHILD_GROUP:
            children.append(record)
        elif group == ADULT_GROUP:
            adults.append(record)
        else:
            ungrouped.append(record)
    return children, adults, ungrouped


def _patient_name(record: dict[str, Any]) -> str:
    """Return the first non-empty patient name available for the encounter."""
    for row in record.get("patient_info", []):
        value = _clean(row.get("patient_name"))
        if value:
            return value
    return ""


def build_label_studio_task(record: dict[str, Any]) -> dict[str, Any]:
    """Wrap an encounter record in Label Studio's import task structure.

    The labeling configs reference ``$text``, ``$patient_id`` and
    ``$patient_name``.  Label Studio resolves those variables from the task's
    ``data`` object, so all encounter fields are retained there and the patient
    name is promoted from ``patient_info`` for display.
    """
    data = deepcopy(record)
    data["patient_name"] = _patient_name(record)
    return {"data": data}


def merge_data(data_dir: Path, main_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and merge all supported CSV files under *data_dir*.

    Returns the encounter records and an audit report.  Rows not reachable from
    the activity table are counted in the report but are not emitted because
    this is intentionally a left join from the activity table.
    """
    files = sorted(data_dir.glob("*.csv"))
    if not main_file.is_absolute():
        main_file = data_dir / main_file
    main_file = main_file.resolve()
    if not main_file.exists():
        raise FileNotFoundError(f"Activity table not found: {main_file}")

    activity_rows = read_csv(main_file)
    missing = {"patient_id", "serial_number"} - set(activity_rows[0] if activity_rows else ())
    if missing:
        raise ValueError(f"Activity table is missing required columns: {sorted(missing)}")

    activity_index = _index_rows(activity_rows, ("patient_id", "serial_number"))
    main_keys = set(activity_index)
    rows_missing_main_key = len(activity_rows) - sum(map(len, activity_index.values()))

    excluded = {main_file}
    patient_file = _find_by_suffix(files, PATIENT_SUFFIX)
    patient_index: dict[tuple[str, ...], list[dict[str, str]]] = {}
    patient_audit: dict[str, Any] | None = None
    if patient_file:
        excluded.add(patient_file.resolve())
        patient_rows = read_csv(patient_file)
        patient_index = _index_rows(patient_rows, ("id",))
        main_patients = {key[0] for key in main_keys}
        matched = sum(
            1 for row in patient_rows if _clean(row.get("id")) in main_patients
        )
        patient_audit = {
            "rows": len(patient_rows),
            "distinct_patient_ids": len(patient_index),
            "matched_rows": matched,
            "unmatched_rows": len(patient_rows) - matched,
        }

    child_files: set[Path] = set()
    parent_child_config: dict[Path, tuple[Path, str]] = {}
    child_audit: dict[str, Any] = {}
    for parent_suffix, (child_suffix, foreign_key) in CHILD_RELATIONSHIPS.items():
        parent_path = _find_by_suffix(files, parent_suffix)
        child_path = _find_by_suffix(files, child_suffix)
        if child_path:
            child_files.add(child_path.resolve())
        if parent_path and child_path:
            parent_child_config[parent_path.resolve()] = (child_path, foreign_key)

    excluded.update(child_files)
    related_indexes: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = {}
    table_audit: dict[str, Any] = {}

    for path in files:
        resolved = path.resolve()
        if resolved in excluded:
            continue
        rows = read_csv(path)
        columns = set(rows[0]) if rows else set()
        if not {"patient_id", "serial_number"}.issubset(columns):
            table_audit[logical_name(path)] = {
                "rows": len(rows),
                "status": "skipped_missing_encounter_key",
            }
            continue

        output_rows: list[dict[str, Any]] = rows
        if resolved in parent_child_config:
            child_path, foreign_key = parent_child_config[resolved]
            child_rows = read_csv(child_path)
            child_index = _index_rows(child_rows, (foreign_key,))
            parent_id_to_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
            parent_groups: dict[
                tuple[tuple[str, str], str], list[dict[str, str]]
            ] = defaultdict(list)
            for position, row in enumerate(rows):
                key = encounter_key(row)
                if key is None:
                    continue
                parent_id = _clean(row.get("id"))
                group_id = parent_id or f"__missing_id_row_{position}"
                parent_groups[(key, group_id)].append(row)
                if parent_id:
                    parent_id_to_keys[parent_id].add(key)

            unique_child_rows = 0
            ambiguous_child_rows = 0
            unmatched_child_rows = 0
            emitted_child_rows = 0
            for child in child_rows:
                foreign_id = _clean(child.get(foreign_key))
                keys = parent_id_to_keys.get(foreign_id, set())
                if not keys:
                    unmatched_child_rows += 1
                elif len(keys) > 1:
                    ambiguous_child_rows += 1
                else:
                    unique_child_rows += 1
                    if next(iter(keys)) in main_keys:
                        emitted_child_rows += 1

            output_rows = []
            for (key, group_id), parent_records in parent_groups.items():
                parent_id = _clean(parent_records[0].get("id"))
                is_unambiguous = len(parent_id_to_keys.get(parent_id, set())) == 1
                output_rows.append(
                    {
                        "patient_id": key[0],
                        "serial_number": key[1],
                        "parent_id": parent_id,
                        "records": deepcopy(parent_records),
                        "items": deepcopy(
                            child_index.get((parent_id,), [])
                            if parent_id and is_unambiguous
                            else []
                        ),
                        "items_status": (
                            "ambiguous_parent_id"
                            if parent_id and not is_unambiguous
                            else "attached"
                        ),
                    }
                )
            child_audit[logical_name(child_path)] = {
                "rows": len(child_rows),
                "foreign_key": foreign_key,
                "matched_rows_to_unique_parent_encounter": unique_child_rows,
                "ambiguous_rows_to_multiple_encounters": ambiguous_child_rows,
                "unmatched_rows_to_parent": unmatched_child_rows,
                "emitted_rows": emitted_child_rows,
            }

        name = logical_name(path)
        related_indexes[name] = _index_rows(output_rows, ("patient_id", "serial_number"))
        table_audit[name] = _table_audit(rows, main_keys)

    records: list[dict[str, Any]] = []
    for patient_id, serial_number in sorted(main_keys):
        key = (patient_id, serial_number)
        record = {
            "patient_id": patient_id,
            "serial_number": serial_number,
            "activity_info": deepcopy(activity_index[key]),
            "patient_info": deepcopy(patient_index.get((patient_id,), [])),
            "encounter_data": {
                name: deepcopy(index.get(key, []))
                for name, index in sorted(related_indexes.items())
            },
        }
        record["age"], record["age_is_approximate"] = calculate_age(record)
        record["age_group"] = classify_age_group(record["age"])
        (
            record["text"],
            record["_text_source_counts"],
            record["_text_duplicate_counts"],
        ) = build_combined_text(record)
        records.append(record)

    text_lengths = [len(record["text"]) for record in records]
    text_source_counts = {
        table: sum(record["_text_source_counts"][table] for record in records)
        for table in TEXT_FIELDS
    }
    text_duplicate_counts = {
        table: sum(record["_text_duplicate_counts"][table] for record in records)
        for table in TEXT_FIELDS
    }
    for record in records:
        del record["_text_source_counts"]
        del record["_text_duplicate_counts"]

    ages = [record["age"] for record in records if record["age"] is not None]

    report = {
        "join": {
            "type": "left",
            "main_table": main_file.name,
            "key": ["patient_id", "serial_number"],
            "output_grain": "one record per distinct composite key",
        },
        "activity": {
            "rows": len(activity_rows),
            "distinct_encounters": len(main_keys),
            "duplicate_key_rows": len(activity_rows) - len(main_keys) - rows_missing_main_key,
            "rows_missing_key": rows_missing_main_key,
        },
        "patient_info": patient_audit,
        "encounter_tables": table_audit,
        "child_tables": child_audit,
        "age": {
            "reference": "earliest activity_time in the encounter",
            "calculated_records": len(ages),
            "exact_records": sum(
                record["age_is_approximate"] is False for record in records
            ),
            "approximate_records": sum(
                record["age_is_approximate"] is True for record in records
            ),
            "missing_records": sum(record["age"] is None for record in records),
            "minimum": min(ages, default=None),
            "maximum": max(ages, default=None),
            "average": round(sum(ages) / len(ages), 2) if ages else None,
            "group_rule": {
                CHILD_GROUP: "age < 18",
                ADULT_GROUP: "age >= 18",
                "未分组": "age is null",
            },
            "group_counts": {
                CHILD_GROUP: sum(
                    record["age_group"] == CHILD_GROUP for record in records
                ),
                ADULT_GROUP: sum(
                    record["age_group"] == ADULT_GROUP for record in records
                ),
                "未分组": sum(record["age_group"] is None for record in records),
            },
        },
        "combined_text": {
            "configured_fields": {
                table: list(fields) for table, fields in TEXT_FIELDS.items()
            },
            "labels": TEXT_LABELS,
            "source_segment_counts": text_source_counts,
            "discarded_exact_duplicates_by_source": text_duplicate_counts,
            "discarded_exact_duplicates": sum(text_duplicate_counts.values()),
            "records_with_text": sum(length > 0 for length in text_lengths),
            "empty_records": sum(length == 0 for length in text_lengths),
            "total_characters": sum(text_lengths),
            "minimum_characters": min(text_lengths, default=0),
            "maximum_characters": max(text_lengths, default=0),
            "average_characters": (
                round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0
            ),
        },
        "output_records": len(records),
    }
    return records, report


def write_tasks_json(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write a Label Studio-compatible JSON list without buffering all tasks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("[\n")
        for index, record in enumerate(records):
            if index:
                handle.write(",\n")
            task = build_label_studio_task(record)
            handle.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n]\n")


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge EMR CSV files into a Label Studio task JSON array."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--main-file", type=Path, default=Path("temp_202608_emr_activity_info.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/emr_merged_202608.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("output/emr_merge_report_202608.json")
    )
    parser.add_argument(
        "--children-output",
        type=Path,
        default=Path("output/emr_merged_202608_children.json"),
    )
    parser.add_argument(
        "--adults-output",
        type=Path,
        default=Path("output/emr_merged_202608_adults.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records, report = merge_data(args.data_dir, args.main_file)
    write_tasks_json(records, args.output)
    children, adults, ungrouped = split_age_groups(records)
    write_tasks_json(children, args.children_output)
    write_tasks_json(adults, args.adults_output)
    write_report(report, args.report)
    print(
        f"Merged {report['activity']['rows']} activity rows into "
        f"{report['output_records']} encounter records: {args.output}"
    )
    print(f"{CHILD_GROUP}: {len(children)} records -> {args.children_output}")
    print(f"{ADULT_GROUP}: {len(adults)} records -> {args.adults_output}")
    if ungrouped:
        print(f"Ungrouped (missing age): {len(ungrouped)} records")
    print(f"Audit report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
