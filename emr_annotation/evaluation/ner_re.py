"""Strict NER and end-to-end directed RE scoring; standard library only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from emr_annotation.evaluation.entity_predictions import (
    build_report, entity_metrics, normalize_records, read_label_groups, render_markdown,
)
from emr_annotation.training_data.frozen import identifier


def relation_keys(row, relation_labels, *, reference):
    """IDs only resolve endpoints; scoring uses character spans and entity types."""
    entities = {}
    for entity in row["entities"]:
        if "id" not in entity:
            continue
        key = identifier(entity["id"])
        if key in entities:
            raise ValueError("Duplicate entity ID in relation evaluation")
        entities[key] = (entity["start"], entity["end"], entity["label"])
    relations = row.get("relations")
    if not isinstance(relations, list):
        raise ValueError("Explicit relations array required, including for negative cases")
    keys = []
    for rel in relations:
        if not isinstance(rel, dict):
            raise ValueError("Relation must be an object")
        source, target = identifier(rel.get("from_id")), identifier(rel.get("to_id"))
        kind, direction = rel.get("type"), rel.get("direction", "right")
        if source not in entities or target not in entities or source == target:
            raise ValueError("Relation endpoints must resolve to distinct entity IDs")
        if kind not in relation_labels or direction not in ("right", "left", "bi"):
            raise ValueError("Missing/unknown relation type or invalid direction")
        pairs = [(source, target)] if direction == "right" else [(target, source)] if direction == "left" else [(source, target), (target, source)]
        keys.extend((entities[a], entities[b], kind) for a, b in pairs)
    if reference and len(keys) != len(set(keys)):
        raise ValueError("Duplicate reference relation")
    return Counter(keys)


def score_relations(reference, predictions, labels):
    counts = {label: Counter() for label in labels}
    by_id = {identifier(row["task_id"]): row for row in predictions}
    if set(by_id) - {identifier(row["task_id"]) for row in reference}:
        raise ValueError("Predictions contain tasks outside reference")
    for gold in reference:
        predicted = by_id.get(identifier(gold["task_id"]))
        g = relation_keys(gold, labels, reference=True)
        p = relation_keys(predicted, labels, reference=False) if predicted and predicted.get("status", "ok") == "ok" else Counter()
        for key, n in (g & p).items():
            counts[key[-1]]["tp"] += n
        for key, n in (p - g).items():
            counts[key[-1]]["fp"] += n
        for key, n in (g - p).items():
            counts[key[-1]]["fn"] += n
    per_label = {label: entity_metrics(*(counts[label][k] for k in ("tp", "fp", "fn"))) for label in labels}
    supported = [m["f1"] for m in per_label.values() if m["support"]]
    return {
        "metric": "directed_relation_type_and_both_endpoint_character_spans_and_entity_labels",
        "per_label": per_label,
        "overall": {
            "micro": entity_metrics(*(sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn"))),
            "macro_f1_supported_labels": sum(supported) / len(supported) if supported else None,
            "supported_label_count": len(supported), "configured_label_count": len(labels),
        },
    }


def joint_report(reference, runs, groups, relation_labels):
    labels = {label for group in groups.values() for label in group}
    gold = normalize_records(reference, labels, reference=True, source="reference")
    normalized = {name: normalize_records(rows, labels, reference=False, source=name) for name, rows in runs.items()}
    if not runs or set(runs) - {"before", "after"}:
        raise ValueError("Expected before and/or after prediction runs")
    report = build_report(gold, normalized.get("before"), groups, after=normalized.get("after"))
    report["relations"] = {name: score_relations(reference, rows, relation_labels) for name, rows in runs.items()}
    report["overall_f1_change"] = {}
    if "before" in runs and "after" in runs:
        for task in ("entities", "relations"):
            scores = report["runs"] if task == "entities" else report["relations"]
            a, b = (scores[name]["overall"]["micro"]["f1"] for name in ("after", "before"))
            report["overall_f1_change"][task] = a - b if a is not None and b is not None else None
    report["notes"].extend([
        "RE is end-to-end on predicted entities; no gold endpoints or sampled negatives are supplied to inference.",
        "Unlabelled directed pairs are negatives; this requires exhaustive adjudicated relation review.",
        "Attributes and case-level choices are not model targets or evaluated here.",
    ])
    return report


def joint_markdown(report):
    def display(value):
        return "—" if value is None else f"{value:.2%}"
    lines = [render_markdown(report), "\n## 端到端关系评估\n",
             "端点跨度、端点实体标签、方向及关系类型必须全部正确。无测试支持的类别以 — 表示。\n",
             "| 运行 | 关系类型 | 参考数 | TP | FP | FN | P | R | F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in report["relations"].items():
        for label, m in {**run["per_label"], "总体 micro": run["overall"]["micro"]}.items():
            lines.append(f"| {name} | {label} | {m['support']} | {m['tp']} | {m['fp']} | {m['fn']} | {display(m['precision'])} | {display(m['recall'])} | {display(m['f1'])} |")
    if report["overall_f1_change"]:
        lines.extend(["\n训练后相对训练前的总体 micro F1 变化（百分点）：\n",
                      *[f"- {key}: {'—' if value is None else f'{value * 100:+.2f}'}" for key, value in report["overall_f1_change"].items()]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("Use a new output directory")
        groups = read_label_groups(args.label_config)
        relation_labels = [node.get("value") for node in ET.parse(args.label_config).getroot().iter("Relation")]
        read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
        runs = {name: read(path) for name, path in (("before", args.before), ("after", args.after)) if path is not None}
        report = joint_report(read(args.reference), runs, groups, relation_labels)
        inputs = {"reference": args.reference, "label_config": args.label_config,
                  **{name: path for name, path in (("before", args.before), ("after", args.after)) if path}}
        report["inputs"] = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in inputs.items()}
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "metrics.md").write_text(joint_markdown(report), encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
