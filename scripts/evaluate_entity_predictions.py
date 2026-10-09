"""Evaluate strict entity matches across regex/before/after runs without a model.

Inputs are normalized task arrays with task_id, exact text, and entities.
Human annotation selection/adjudication must happen before this evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


Entity = tuple[int, int, str]


@dataclass(frozen=True)
class Record:
    task_id: str
    text: str
    entities: tuple[Entity, ...]
    status: str = "ok"


def read_label_groups(path: Path) -> dict[str, list[str]]:
    """Only entity Labels controls count; Choices/relations are separate tasks."""
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for control in ET.parse(path).getroot().iter():
        if control.tag.rsplit("}", 1)[-1] != "Labels":
            continue
        name = control.attrib.get("name", "")
        if not name or name in groups:
            raise ValueError("Entity controls require distinct nonempty names")
        labels = []
        for child in control:
            if child.tag.rsplit("}", 1)[-1] != "Label":
                continue
            label = child.attrib.get("value", "")
            if not label or label in seen:
                raise ValueError("Entity label values must be nonempty and unique")
            seen.add(label)
            labels.append(label)
        groups[name] = labels
    if not seen:
        raise ValueError("No entity Labels found in label configuration")
    return groups


def normalize_records(
    payload: Any, labels: set[str], *, reference: bool, source: str
) -> dict[str, Record]:
    if not isinstance(payload, list):
        raise ValueError(f"{source}: expected a normalized task array")
    records = {}
    for index, task in enumerate(payload):
        where = f"{source}: task index {index}"
        if not isinstance(task, dict):
            raise ValueError(f"{where}: expected an object")
        task_id = task.get("task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, (str, int))
            or not str(task_id).strip()
        ):
            raise ValueError(f"{where}: task_id must be a nonempty string or integer")
        task_id = str(task_id)
        if task_id in records:
            raise ValueError(f"{where}: duplicate task_id")
        text = task.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{where}: exact original text is required")
        status = task.get("status", "ok")
        if not isinstance(status, str) or status not in {"ok", "failed"} or (reference and status != "ok"):
            raise ValueError(f"{where}: invalid status for this input")
        raw_entities = task.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError(f"{where}: explicit entities array is required")
        if status == "failed" and raw_entities:
            raise ValueError(f"{where}: failed predictions must have no entities")
        entities = []
        for entity in raw_entities:
            if not isinstance(entity, dict):
                raise ValueError(f"{where}: entity must be an object")
            start, end, label = (
                entity.get("start"), entity.get("end"), entity.get("label")
            )
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(text)
            ):
                raise ValueError(f"{where}: entity offsets are invalid")
            if not isinstance(label, str) or label not in labels:
                raise ValueError(f"{where}: entity label is outside the configuration")
            if "text" in entity and entity["text"] != text[start:end]:
                raise ValueError(f"{where}: entity text differs from its original span")
            entities.append((start, end, label))
        if reference and len(set(entities)) != len(entities):
            raise ValueError(f"{where}: duplicate reference entity")
        records[task_id] = Record(task_id, text, tuple(entities), status)
    if reference and not records:
        raise ValueError(f"{source}: reference set must contain at least one task")
    return records


def entity_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    support = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "predicted": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / support if support else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if support else None,
        "status": "evaluated" if support else "no_reference_examples",
    }


def score_run(
    reference: dict[str, Record],
    predictions: dict[str, Record],
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    extra = set(predictions) - set(reference)
    if extra:
        raise ValueError("Prediction run contains tasks outside the reference set")
    labels = [label for group in groups.values() for label in group]
    counts = {label: Counter() for label in labels}
    presence = {label: Counter() for label in labels}
    missing, failed = [], []
    duplicate_predictions = 0
    for task_id, gold in reference.items():
        predicted = predictions.get(task_id)
        if predicted is not None and predicted.text != gold.text:
            raise ValueError("Prediction text differs from the exact reference text")
        if predicted is None:
            missing.append(task_id)
        elif predicted.status == "failed":
            failed.append(task_id)
        valid = predicted is not None and predicted.status == "ok"
        gold_entities = Counter(gold.entities)
        pred_entities = Counter(predicted.entities if valid else ())
        duplicate_predictions += sum(n - 1 for n in pred_entities.values())
        # Counter intersection enforces one-to-one matching. Repeated output is FP.
        for (_, _, label), number in (gold_entities & pred_entities).items():
            counts[label]["tp"] += number
        for (_, _, label), number in (pred_entities - gold_entities).items():
            counts[label]["fp"] += number
        for (_, _, label), number in (gold_entities - pred_entities).items():
            counts[label]["fn"] += number
        gold_labels = {entity[2] for entity in gold.entities}
        pred_labels = {entity[2] for entity in pred_entities}
        for label in labels:
            if label in gold_labels:
                presence[label]["tp" if label in pred_labels else "fn"] += 1
            elif not valid:
                # An absent/failed response never earns credit for a negative case.
                presence[label]["failed_negative"] += 1
            else:
                presence[label]["fp" if label in pred_labels else "tn"] += 1
    per_label = {
        label: entity_metrics(counts[label]["tp"], counts[label]["fp"], counts[label]["fn"])
        for label in labels
    }

    def aggregate(selected: list[str]) -> dict[str, Any]:
        micro = entity_metrics(
            *(sum(counts[label][key] for label in selected) for key in ("tp", "fp", "fn"))
        )
        supported = [per_label[label]["f1"] for label in selected if per_label[label]["support"]]
        return {
            "micro": micro,
            "macro_f1_supported_labels": sum(supported) / len(supported) if supported else None,
            "supported_label_count": len(supported),
            "configured_label_count": len(selected),
        }

    per_label_presence = {}
    for label in labels:
        c = presence[label]
        per_label_presence[label] = {
            key: c[key] for key in ("tp", "fp", "fn", "tn", "failed_negative")
        }
        per_label_presence[label].update({
            "reference_positive_tasks": c["tp"] + c["fn"],
            "total_tasks": len(reference),
            "accuracy": (c["tp"] + c["tn"]) / len(reference),
            "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
            "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
        })
    return {
        "coverage": {
            "reference_tasks": len(reference),
            "successful_prediction_tasks": len(reference) - len(missing) - len(failed),
            "missing_prediction_tasks": len(missing),
            "failed_prediction_tasks": len(failed),
            "missing_task_ids": missing,
            "failed_task_ids": failed,
            "duplicate_prediction_entities": duplicate_predictions,
        },
        "overall": aggregate(labels),
        "groups": {name: aggregate(group) for name, group in groups.items()},
        "per_label": per_label,
        "label_mention_presence": per_label_presence,
    }


def build_report(
    reference: dict[str, Record],
    before: dict[str, Record] | None,
    groups: dict[str, list[str]],
    after: dict[str, Record] | None = None,
    regex: dict[str, Record] | None = None,
) -> dict[str, Any]:
    report = {
        "protocol_version": 2,
        "metric": "strict_character_span_and_entity_label",
        "notes": [
            "Inputs must use adjudicated reference entities and identical original text.",
            "Patient separation, adjudication and model versions are not verified by this scorer.",
            "Missing/failed predictions remain in the denominator and produce missed entities.",
            "Labels with no reference examples have null recall/F1; false positives are retained.",
            "Macro F1 averages only labels with reference examples; micro includes all false positives.",
            "Mention presence measures label mentions, not clinical symptom existence or attributes.",
            "Missing/failed negative cases count as errors in mention-presence accuracy.",
        ],
        "label_groups": groups,
        "runs": {},
    }
    for name, predictions in (("regex", regex), ("before", before), ("after", after)):
        if predictions is not None:
            report["runs"][name] = score_run(reference, predictions, groups)
    if not report["runs"]:
        raise ValueError("At least one prediction run is required")
    if before is not None and after is not None:
        report["f1_change"] = {}
        for label, metrics in report["runs"]["before"]["per_label"].items():
            a = report["runs"]["after"]["per_label"][label]["f1"]
            b = metrics["f1"]
            report["f1_change"][label] = a - b if a is not None and b is not None else None
    report["comparisons"] = {}
    for candidate, baseline in (("before", "regex"), ("after", "regex"), ("after", "before")):
        if candidate not in report["runs"] or baseline not in report["runs"]:
            continue
        comparison = {}
        for label, old in report["runs"][baseline]["per_label"].items():
            new = report["runs"][candidate]["per_label"][label]
            changes = {}
            for metric in ("precision", "recall", "f1"):
                changes[metric + "_change"] = new[metric] - old[metric] if new[metric] is not None and old[metric] is not None else None
            changes["mention_accuracy_change"] = (
                report["runs"][candidate]["label_mention_presence"][label]["accuracy"]
                - report["runs"][baseline]["label_mention_presence"][label]["accuracy"]
            )
            comparison[label] = changes
        report["comparisons"][f"{candidate}_vs_{baseline}"] = comparison
    return report


def render_markdown(report: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    runs = report["runs"]
    lines = [
        "# 实体抽取逐标签评价" + ("（虚构数据演示）" if report.get("dataset_kind") == "synthetic_demo" else ""),
        "",
        "口径：原文字符跨度和实体标签均精确匹配。无参考样本的标签以“—”显示召回率和F1，仍统计误报。",
        "参考是否经过医学裁决、测试患者是否独立、规则是否冻结、模型是否确为训练前后版本，需要另附记录核实。",
        "",
        "## 任务覆盖与总体结果",
        "",
        "| 运行 | 参考任务 | 成功预测 | 缺失 | 失败 | 重复预测实体 | micro P | micro R | micro F1 | macro F1（有参考类） | 有参考类/配置类 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        c, m = run["coverage"], run["overall"]["micro"]
        lines.append(
            f"| {name} | {c['reference_tasks']} | {c['successful_prediction_tasks']} | "
            f"{c['missing_prediction_tasks']} | {c['failed_prediction_tasks']} | "
            f"{c['duplicate_prediction_entities']} | {percent(m['precision'])} | "
            f"{percent(m['recall'])} | {percent(m['f1'])} | "
            f"{percent(run['overall']['macro_f1_supported_labels'])} | "
            f"{run['overall']['supported_label_count']}/{run['overall']['configured_label_count']} |"
        )
    for group, labels in report["label_groups"].items():
        safe_group = group.replace("|", "\\|").replace("\n", " ")
        lines.extend(["", f"## {safe_group}", ""])
        header = "| 标签 | 参考实体数 |"
        separator = "|---|---:|"
        for name in runs:
            header += f" {name} TP/FP/FN | {name} P/R/F1 |"
            separator += "---|---|"
        for name in report.get("comparisons", {}):
            header += f" {name} F1变化（百分点） |"
            separator += "---:|"
        lines.extend([header, separator])
        for label in labels:
            safe_label = label.replace("|", "\\|").replace("\n", " ")
            support = next(iter(runs.values()))["per_label"][label]["support"]
            line = f"| {safe_label} | {support} |"
            for run in runs.values():
                m = run["per_label"][label]
                line += f" {m['tp']}/{m['fp']}/{m['fn']} | "
                line += "/".join(percent(m[k]) for k in ("precision", "recall", "f1")) + " |"
            for comparison in report.get("comparisons", {}).values():
                delta = comparison[label]["f1_change"]
                line += f" {100 * delta:+.1f} |" if delta is not None else " — |"
            lines.append(line)
    lines.extend([
        "",
        "## 标签提及是否出现：辅助准确率",
        "",
        "此表仅判断病历是否包含相应标签提及，不衡量跨度定位、否定状态或患者当前是否有该症状。缺失/失败响应计为错误。",
        "",
        "| 标签 | 含该标签的参考任务数 |" + "".join(f" {name} 准确率 |" for name in runs),
        "|---|---:|" + "---:|" * len(runs),
    ])
    for label, b in next(iter(runs.values()))["label_mention_presence"].items():
        safe_label = label.replace("|", "\\|").replace("\n", " ")
        line = f"| {safe_label} | {b['reference_positive_tasks']} |"
        for run in runs.values():
            line += f" {percent(run['label_mention_presence'][label]['accuracy'])} |"
        lines.append(line)
    lines.extend(["", "完整原始计数、缺失与失败任务标识及输入校验值见配套JSON。此报告不评价关系、属性、病例判别或运行速度。", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--regex", type=Path)
    parser.add_argument("--dataset-kind", choices=("unverified", "synthetic_demo", "human_test_set"), default="unverified")
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = {"reference": args.reference, "label_config": args.label_config}
        for name in ("regex", "before", "after"):
            if getattr(args, name) is not None:
                inputs[name] = getattr(args, name)
        outputs = [p for p in (args.output, args.markdown) if p is not None]
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError("JSON and Markdown outputs must have different paths")
        if {p.resolve() for p in outputs} & {p.resolve() for p in inputs.values()}:
            raise ValueError("Output must not overwrite an input")
        groups = read_label_groups(args.label_config)
        labels = {label for group in groups.values() for label in group}

        def read_records(path: Path, reference: bool) -> dict[str, Record]:
            return normalize_records(
                json.loads(path.read_text(encoding="utf-8-sig")), labels,
                reference=reference, source=path.name,
            )

        report = build_report(
            read_records(args.reference, True), read_records(args.before, False) if args.before is not None else None, groups,
            read_records(args.after, False) if args.after is not None else None,
            read_records(args.regex, False) if args.regex is not None else None,
        )
        report["dataset_kind"] = args.dataset_kind
        report["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["inputs"] = {
            name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.markdown is not None:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(render_markdown(report), encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    count = next(iter(report["runs"].values()))["coverage"]["reference_tasks"]
    print(f"Evaluated {len(report['runs'])} run(s); reference tasks={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
