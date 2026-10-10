"""Audit a local Label Studio JSON export and explicitly select reviewed references.

No network access, model inference, automatic adjudication or deidentification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from emr_annotation.annotation.schema import load_export_schema as load_schema

from emr_annotation.evaluation.entity_predictions import normalize_records, read_label_groups


def identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Identifier must be a nonempty string or integer")
    return str(value)




def parse_annotation(annotation: dict[str, Any], text: str, schema: dict[str, Any]) -> dict[str, Any]:
    results = annotation.get("result")
    if not isinstance(results, list) or not results:
        raise ValueError("Empty or invalid result; not automatically a reviewed negative")
    entities, attributes, relations = [], [], []
    region_keys: dict[str, tuple] = {}
    case_choices = {}
    pending_attributes, pending_relations = [], []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Result entry must be an object")
        kind = result.get("type")
        if kind == "relation":
            pending_relations.append(result)
            continue
        if result.get("to_name") != schema["text_control"]:
            raise ValueError("Result is attached to a different Text control")
        value = result.get("value")
        if not isinstance(value, dict):
            raise ValueError("Result value must be an object")
        name = result.get("from_name")
        if not isinstance(name, str) or not name:
            raise ValueError("Result requires a named annotation control")
        if kind == "labels":
            if name not in schema["groups"]:
                raise ValueError("Unknown entity control")
            labels = value.get("labels")
            if not isinstance(labels, list) or not labels or any(
                not isinstance(label, str) or label not in schema["groups"][name] for label in labels
            ):
                raise ValueError("Entity label does not belong to its configured control")
            if schema.get("entity_choices", {}).get(name, "single") == "single" and len(labels) != 1:
                raise ValueError("Single-selection entity control requires exactly one label")
            region_id = identifier(result.get("id"))
            if region_id in region_keys:
                raise ValueError("Repeated entity region ID")
            for label in labels:
                entity = {"id": region_id, "start": value.get("start"), "end": value.get("end"), "label": label}
                if "text" in value:
                    entity["text"] = value["text"]
                entities.append(entity)
            region_keys[region_id] = (value.get("start"), value.get("end"), tuple(sorted(labels)))
        elif kind == "choices":
            if name not in schema["choices"]:
                raise ValueError("Unknown Choices control")
            values = value.get("choices")
            if not isinstance(values, list) or len(values) != 1 or any(
                not isinstance(v, str) or v not in schema["choices"][name]["value_map"] for v in values
            ):
                raise ValueError("Expected one configured choice value")
            normalized_values = [schema["choices"][name]["value_map"][v] for v in values]
            if schema["choices"][name]["per_region"]:
                pending_attributes.append({"region_id": identifier(result.get("id")), "name": name,
                                           "values": normalized_values, "export_values": values})
            elif name in case_choices:
                raise ValueError("Repeated case-level choice")
            else:
                case_choices[name] = normalized_values[0]
        else:
            raise ValueError("Unsupported result type in this text-annotation converter")
    labels = {label for group in schema["groups"].values() for label in group}
    normalize_records([{"task_id": "validation", "text": text, "entities": entities}], labels, reference=True, source="annotation")
    seen_attributes = set()
    for attr in pending_attributes:
        key = (attr["region_id"], attr["name"])
        if attr["region_id"] not in region_keys or key in seen_attributes:
            raise ValueError("Attribute has a missing entity region or repeats a field")
        seen_attributes.add(key)
        attributes.append(attr)
    for relation in pending_relations:
        source, target = identifier(relation.get("from_id")), identifier(relation.get("to_id"))
        values = relation.get("labels")
        if source not in region_keys or target not in region_keys:
            raise ValueError("Relation refers to a missing entity region")
        if not isinstance(values, list) or len(values) != 1 or any(
            not isinstance(v, str) or v not in schema["relations"] for v in values
        ):
            raise ValueError("Expected one configured relation label")
        direction = relation.get("direction", "right")
        if not isinstance(direction, str) or direction not in {"left", "right", "bi"}:
            raise ValueError("Unsupported relation direction")
        relations.append({"from_id": source, "to_id": target, "type": values[0], "direction": direction})
    for name, control in schema["choices"].items():
        if not control["per_region"] and control["required"] and name not in case_choices:
            raise ValueError("Required case-level choice is missing")

    # Compare semantic spans, not independently generated region IDs.
    signatures = {
        "entities": sorted((e["start"], e["end"], e["label"]) for e in entities),
        "attributes": sorted((region_keys[a["region_id"]], a["name"], tuple(a["values"])) for a in attributes),
        "relations": sorted((region_keys[r["from_id"]], region_keys[r["to_id"]], r["type"], r["direction"]) for r in relations),
        "case_choices": sorted(case_choices.items()),
    }
    return {"entities": entities, "attributes": attributes, "relations": relations, "case_choices": case_choices, "signatures": signatures}


def audit_export(payload: Any, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a nonempty Label Studio JSON task array")
    tasks, details = {}, []
    totals = Counter()
    agreement = Counter()
    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"Task index {index} is not an object")
        task_id = identifier(task.get("id"))
        if task_id in tasks:
            raise ValueError("Repeated task ID in export")
        data = task.get("data")
        if not isinstance(data, dict) or not isinstance(data.get(schema["text_field"]), str):
            raise ValueError(f"Task index {index} lacks the configured original text")
        text = data[schema["text_field"]]
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("Task annotations must be an array")
        valid, annotations_audit = {}, []
        seen_ids = set()
        for annotation in annotations:
            totals["annotations"] += 1
            summary = {"annotation_id": None, "annotator_id": None, "status": "invalid"}
            try:
                if not isinstance(annotation, dict):
                    raise ValueError("Annotation must be an object")
                ann_id = identifier(annotation.get("id"))
                summary["annotation_id"] = ann_id
                if ann_id in seen_ids:
                    # Ambiguous IDs invalidate both copies, including a previously valid one.
                    if valid.pop(ann_id, None) is not None:
                        totals["valid_annotations"] -= 1
                        totals["invalid_annotations"] += 1
                        for prior in annotations_audit:
                            if prior["annotation_id"] == ann_id and prior["status"] == "valid":
                                prior.update({"status": "invalid", "error": "Repeated annotation ID within a task"})
                    raise ValueError("Repeated annotation ID within a task")
                seen_ids.add(ann_id)
                cancelled = annotation.get("was_cancelled", False)
                if not isinstance(cancelled, bool):
                    raise ValueError("was_cancelled must be boolean")
                if cancelled:
                    summary["status"] = "cancelled"
                    totals["cancelled_annotations"] += 1
                    annotations_audit.append(summary)
                    continue
                annotator = annotation.get("completed_by")
                if isinstance(annotator, dict):
                    annotator = annotator.get("id")
                summary["annotator_id"] = identifier(annotator) if annotator is not None else None
                parsed = parse_annotation(annotation, text, schema)
                valid[ann_id] = {**parsed, "annotator_id": summary["annotator_id"]}
                summary.update({"status": "valid", "entity_count": len(parsed["entities"]), "case_choices": parsed["case_choices"]})
                totals["valid_annotations"] += 1
            except ValueError as exc:
                summary["error"] = str(exc)
                totals["invalid_annotations"] += 1
            annotations_audit.append(summary)
        identities = [annotation["annotator_id"] for annotation in valid.values()]
        pair_eligible = len(valid) == 2 and None not in identities and len(set(identities)) == 2
        matches = None
        if pair_eligible:
            first, second = list(valid.values())
            matches = {key: first["signatures"][key] == second["signatures"][key] for key in first["signatures"]}
            totals["two_distinct_annotator_tasks"] += 1
            for key, same in matches.items():
                agreement[key] += int(same)
        if len(valid) != 2:
            totals["tasks_without_exactly_two_valid_annotations"] += 1
        elif not pair_eligible:
            totals["two_annotations_but_identity_unverified_or_same"] += 1
        patient = data.get("patient_id")
        patient_id = identifier(patient) if patient is not None and str(patient).strip() else None
        if patient_id is None:
            totals["tasks_missing_patient_id"] += 1
        tasks[task_id] = {"text": text, "patient_id": patient_id, "annotations": valid}
        details.append({"task_id": task_id, "valid_annotation_ids": list(valid), "annotations": annotations_audit, "pair_eligible": pair_eligible, "pair_exact_matches": matches})
    pair_count = totals["two_distinct_annotator_tasks"]
    report = {
        "summary": {"tasks": len(tasks), **dict(totals)},
        "pair_exact_agreement": {
            key: {"matching_tasks": agreement[key], "comparable_tasks": pair_count, "rate": agreement[key] / pair_count if pair_count else None}
            for key in ("entities", "attributes", "relations", "case_choices")
        },
        "notes": [
            "Counts describe this export only, not verified independent human annotation or clinical validity.",
            "Pairs require exactly two valid annotations with distinct known annotator IDs.",
            "Exact set agreement is descriptive, not entity F1 or Cohen kappa.",
            "Conditional required attributes and full Label Studio UI behavior are not validated.",
            "Platform predictions are never selected as human reference.",
        ],
        "tasks": details,
    }
    return report, tasks


def select_reference(tasks: dict[str, Any], selection: Any, provenance: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(selection, dict) or any(selection.get(key) != value for key, value in provenance.items()):
        raise ValueError("Selection source/config hashes do not match this export and configuration")
    rows = selection.get("selections")
    if not isinstance(rows, list):
        raise ValueError("Selection requires a selections array")
    choices = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Selection row must be an object")
        task_id = identifier(row.get("task_id"))
        if task_id in choices:
            raise ValueError("Repeated selection task ID")
        choices[task_id] = row
    if set(choices) != set(tasks):
        raise ValueError("Selection must explicitly account for every exported task")
    reference, included, excluded = [], [], []
    for task_id, task in tasks.items():
        choice = choices[task_id]
        if choice.get("reviewed") is not True or not isinstance(choice.get("reviewer"), str) or not choice["reviewer"].strip():
            raise ValueError("Every inclusion/exclusion requires reviewed=true and a reviewer")
        action = choice.get("action")
        if action == "exclude":
            if not isinstance(choice.get("reason"), str) or not choice["reason"].strip():
                raise ValueError("Excluded tasks require a recorded reason")
            excluded.append({"task_id": task_id, "reviewer": choice["reviewer"], "reason": choice["reason"]})
            continue
        if action != "include":
            raise ValueError("Pending tasks must be reviewed before generating reference")
        ann_id = identifier(choice.get("annotation_id"))
        if ann_id not in task["annotations"]:
            raise ValueError("Selected annotation is invalid, cancelled or absent from this export")
        annotation = task["annotations"][ann_id]
        reference.append({
            "task_id": task_id, "text": task["text"], "patient_id": task["patient_id"],
            "entities": annotation["entities"], "attributes": annotation["attributes"],
            "relations": annotation["relations"], "case_choices": annotation["case_choices"],
            "source_annotation_id": ann_id,
        })
        included.append({"task_id": task_id, "annotation_id": ann_id, "reviewer": choice["reviewer"], "reason": choice.get("reason", "")})
    if not reference:
        raise ValueError("Selection must include at least one reviewed task")
    return reference, {**provenance, "included": included, "excluded": excluded}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.export.read_bytes()
        provenance = {
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "label_config_sha256": hashlib.sha256(args.label_config.read_bytes()).hexdigest(),
        }
        schema = load_schema(args.label_config)
        audit, tasks = audit_export(json.loads(raw.decode("utf-8-sig")), schema)
        audit.update(provenance)
        outputs: dict[str, Any] = {"audit.json": audit}
        if args.selection is None:
            outputs["selection_template.json"] = {
                **provenance,
                "selections": [
                    {"task_id": task_id, "action": "pending", "annotation_id": None, "reviewed": False, "reviewer": "", "reason": ""}
                    for task_id in tasks
                ],
            }
        else:
            selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
            reference, receipt = select_reference(tasks, selection, provenance)
            outputs.update({"reference.json": reference, "selection_receipt.json": receipt})
        input_paths = {p.resolve() for p in (args.export, args.label_config, args.selection) if p is not None}
        for name in outputs:
            path = args.output_dir / name
            if path.exists() or path.resolve() in input_paths:
                raise ValueError("Use a fresh output directory; existing inputs or results are never overwritten")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in outputs.items():
            (args.output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"Audited {len(tasks)} task(s); valid annotations={audit['summary'].get('valid_annotations', 0)}")
    print("Reviewed reference generated" if args.selection is not None else "Review required; no reference generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
