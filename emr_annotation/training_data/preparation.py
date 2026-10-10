"""Convert a full Label Studio JSON export to reviewed NER+RE training files.

Without --selection, write an audit and pending selection template only.
With --selection, convert explicitly reviewed annotations and optionally split
by patient/exact text. No model, training, network access or auto-adjudication.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from emr_annotation.annotation_analysis.label_studio_evaluation import audit_export, load_schema, select_reference
from emr_annotation.training_data.frozen import convert_rows, prepare_frozen_training
from emr_annotation.training_data.split_reference import split_reference


def json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def convert_offset_units(payload, unit):
    """Explicit unit conversion, never text rewriting or heuristic offset repair."""
    if unit == "codepoint":
        return payload
    if unit != "utf16" or not isinstance(payload, list):
        raise ValueError("Expected task array and offset unit codepoint or utf16")
    result = copy.deepcopy(payload)
    for task in result:
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict) or not isinstance(task["data"].get("text"), str):
            raise ValueError("UTF-16 conversion requires data.text on every task")
        boundaries, position = {0: 0}, 0
        for index, char in enumerate(task["data"]["text"]):
            position += 2 if ord(char) > 0xFFFF else 1
            boundaries[position] = index + 1
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("annotations must be an array")
        for annotation in annotations:
            if not isinstance(annotation, dict) or not isinstance(annotation.get("result"), list):
                continue  # audit_export records invalid annotations.
            for item in annotation["result"]:
                if not isinstance(item, dict) or item.get("type") != "labels" or not isinstance(item.get("value"), dict):
                    continue
                value = item["value"]
                for key in ("start", "end"):
                    offset = value.get(key)
                    if type(offset) is not int or offset not in boundaries:
                        # Leave an invalid value for the annotation auditor; never guess.
                        value[key] = None
                    else:
                        value[key] = boundaries[offset]
    return result


def build_outputs(export_bytes, label_config, selection=None, ratios=None, seed=42, offset_unit="codepoint", dataset_kind="unverified"):
    schema = load_schema(label_config)
    if schema["text_field"] != "text":
        raise ValueError("This training pipeline expects the project data.text contract")
    provenance = {"source_sha256": hashlib.sha256(export_bytes).hexdigest(),
                  "label_config_sha256": hashlib.sha256(label_config.read_bytes()).hexdigest()}
    raw = json.loads(export_bytes.decode("utf-8-sig"))
    audit, tasks = audit_export(convert_offset_units(raw, offset_unit), schema)
    audit.update(provenance)
    outputs = {"audit.json": audit}
    report = {"protocol_version": 1, "dataset_kind": dataset_kind, "offset_unit_in": offset_unit,
              "offset_unit_out": "python_codepoint", "source_sha256": provenance["source_sha256"],
              "label_config_sha256": provenance["label_config_sha256"], "export_tasks": len(tasks),
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "notes": ["The input export is never modified.", "Predictions, cancelled and invalid annotations cannot be selected as reference.",
                        "Choice aliases normalize to display values; original per-region choice strings remain in export_values.",
                        "Case choices and attributes are preserved in reference.json but are not learned by the existing NER+RE dataset.",
                        "training_data.json covers all selected records and is not an independent test set.",
                        "No model training, tokenizer alignment, UI verification or medical adjudication is performed."]}
    if selection is None:
        if ratios is not None:
            raise ValueError("Provide reviewed --selection before producing splits")
        outputs["selection_template.json"] = {**provenance, "offset_unit_in": offset_unit, "selections": [
            {"task_id": task_id, "action": "pending", "annotation_id": None, "reviewed": False, "reviewer": "", "reason": ""}
            for task_id in tasks]}
        report["status"] = "audit_only_review_required"
    else:
        if not isinstance(selection, dict) or selection.get("offset_unit_in", "codepoint") != offset_unit:
            raise ValueError("Selection offset_unit_in must match --offset-unit; regenerate the audit template")
        reference, receipt = select_reference(tasks, selection, provenance)
        labels = {v for values in schema["groups"].values() for v in values}
        relations = [node.get("value") for node in ET.parse(label_config).getroot().iter() if node.tag.rsplit("}",1)[-1] == "Relation" and node.get("value")]
        training = convert_rows(reference, labels, relations)
        outputs.update({"reference.json": reference, "selection_receipt.json": receipt, "training_data.json": training})
        report.update({"status": "reviewed_training_data_converted", "included_tasks": len(reference),
                       "excluded_tasks": len(receipt["excluded"]), "no_entity_tasks": sum(not r["entities"] for r in reference),
                       "entities": sum(len(r["entities"]) for r in reference), "relations_in_model_data": sum(len(r["relations"]) for r in training)})
        if ratios is not None:
            splits, manifest = split_reference(reference, schema["groups"], tuple(ratios), seed)
            manifest.update({"source_sha256": hashlib.sha256(json_bytes(reference)).hexdigest(),
                             "label_config_sha256": provenance["label_config_sha256"]})
            outputs["splits/split_manifest.json"] = manifest
            for name, rows in splits.items():
                outputs[f"splits/{name}.json"] = rows
                outputs[f"model_data/{name}.json"] = convert_rows(rows, labels, relations)
            report["split_tasks"] = {name: len(rows) for name, rows in splits.items()}
    outputs["conversion_report.json"] = report
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset-unit", choices=("codepoint", "utf16"), default="codepoint")
    parser.add_argument("--dataset-kind", choices=("unverified", "synthetic_demo", "human_reviewed"), default="unverified")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("Use a new output directory; previous inputs/results are never overwritten")
        selection = json.loads(args.selection.read_text(encoding="utf-8-sig")) if args.selection else None
        outputs = build_outputs(args.export.read_bytes(), args.label_config, selection, args.ratios, args.seed, args.offset_unit, args.dataset_kind)
        report = outputs["conversion_report.json"]
        report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["selection_sha256"] = hashlib.sha256(args.selection.read_bytes()).hexdigest() if args.selection else None
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete validated directory; no partial successful conversion.
        with tempfile.TemporaryDirectory(prefix=".ls-training-", dir=args.output_dir.parent) as temp:
            stage = Path(temp) / "result"
            stage.mkdir()
            for name, payload in outputs.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(json_bytes(payload))
            if "splits/split_manifest.json" in outputs:
                _, _, preflight = prepare_frozen_training(stage / "splits", stage / "reference.json", args.label_config)
                (stage / "frozen_training_preflight.json").write_bytes(json_bytes(preflight))
            stage.rename(args.output_dir)
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"{report['status']}; exported tasks={report['export_tasks']}; dataset={args.dataset_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
