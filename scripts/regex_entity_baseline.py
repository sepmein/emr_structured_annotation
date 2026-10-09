"""Run a versioned regex entity baseline and time local CPU extraction.

Only task_id and unchanged text are read from tasks; reference entities are ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import time
from typing import Any
import xml.etree.ElementTree as ET

if __package__:
    from .evaluate_entity_predictions import normalize_records, read_label_groups
else:
    from evaluate_entity_predictions import normalize_records, read_label_groups


@dataclass(frozen=True)
class Rule:
    label: str
    control: str
    patterns: tuple[re.Pattern, ...]
    priority: int
    exclude_before: re.Pattern | None
    exclude_after: re.Pattern | None


def compile_rules(payload: Any, groups: dict[str, list[str]]) -> list[Rule]:
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str) or not payload["version"].strip():
        raise ValueError("Rules require an explicit nonempty version")
    entries = payload.get("rules")
    if not isinstance(entries, list):
        raise ValueError("Rules must contain a rules array")
    controls = {label: control for control, labels in groups.items() for label in labels}
    seen, result = set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Rule must be an object")
        label = entry.get("label")
        if not isinstance(label, str) or label not in controls or label in seen:
            raise ValueError("Rule labels must occur exactly once and match the XML")
        seen.add(label)
        patterns = entry.get("patterns")
        if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) or not p for p in patterns):
            raise ValueError("Every configured label requires a nonempty regex pattern list")
        compiled = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
        if any((match := pattern.search("")) is not None and match.start() == match.end() for pattern in compiled):
            raise ValueError("Regex patterns must not produce empty spans")
        priority = entry.get("priority", 10)
        if type(priority) is not int:
            raise ValueError("Rule priority must be an integer")
        exclusions = []
        for key in ("exclude_before", "exclude_after"):
            pattern = entry.get(key)
            if pattern is not None and (not isinstance(pattern, str) or not pattern):
                raise ValueError("Context exclusion must be a nonempty regex string")
            exclusions.append(re.compile(pattern, re.IGNORECASE) if pattern else None)
        result.append(Rule(label, controls[label], compiled, priority, *exclusions))
    if seen != set(controls):
        raise ValueError("Rules must explicitly cover every XML entity label")
    return result


def predict_entities(text: str, rules: list[Rule]) -> list[dict[str, Any]]:
    candidates = {}
    for rule in rules:
        for pattern in rule.patterns:
            for match in pattern.finditer(text):
                start, end = match.span("entity") if "entity" in pattern.groupindex else match.span()
                if start < 0 or start >= end:
                    raise ValueError("Regex produced an empty or unmatched entity capture")
                if rule.exclude_before and rule.exclude_before.search(text[max(0, start - 24):start]):
                    continue
                if rule.exclude_after and rule.exclude_after.search(text[end:end + 24]):
                    continue
                key = (start, end, rule.label)
                candidates[key] = (rule.priority, rule.control)
    accepted, occupied = [], {}
    for (start, end, label), (priority, control) in sorted(
        candidates.items(), key=lambda item: (item[1][0], -(item[0][1] - item[0][0]), item[0])
    ):
        if any(start < old_end and end > old_start for old_start, old_end in occupied.get(control, [])):
            continue
        occupied.setdefault(control, []).append((start, end))
        accepted.append({"start": start, "end": end, "label": label, "text": text[start:end]})
    return sorted(accepted, key=lambda entity: (entity["start"], entity["end"], entity["label"]))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def run_baseline(payload: Any, groups: dict[str, list[str]], rules: list[Rule], repeats: int, warmup: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("Tasks must be a normalized task array")
    # Deliberately do not read gold entities, choices, patient IDs or annotation metadata.
    inference_inputs = [{"task_id": row.get("task_id"), "text": row.get("text"), "entities": []} for row in payload]
    labels = {label for group in groups.values() for label in group}
    tasks = normalize_records(inference_inputs, labels, reference=True, source="tasks")
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    for _ in range(warmup):
        for task in tasks.values():
            predict_entities(task.text, rules)
    predictions, samples, wall_seconds = [], [], 0.0
    for repeat in range(repeats):
        wall_start = time.perf_counter_ns()
        for task in tasks.values():
            start = time.perf_counter_ns()
            entities = predict_entities(task.text, rules)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            samples.append({"task_id": task.task_id, "repeat": repeat, "latency_ms": elapsed, "text_chars": len(task.text), "entity_count": len(entities), "status": "ok"})
            if repeat == 0:
                predictions.append({"task_id": task.task_id, "text": task.text, "entities": entities, "status": "ok"})
        wall_seconds += (time.perf_counter_ns() - wall_start) / 1_000_000_000
    latencies = [sample["latency_ms"] for sample in samples]
    return predictions, {
        "method": "regex", "measurement_scope": "local_cpu_entity_extraction",
        "notes": [
            "Per-task latency includes regex matching, exclusions, overlap resolution and entity construction.",
            "Input/output file I/O, validation, compilation and warmup are excluded.",
            "Throughput uses total measured loop wall time, including timing bookkeeping.",
            "These local timings are not directly comparable with model HTTP/queue latency.",
            "Text length is recorded; regex scans the entire original text without truncation.",
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or "unavailable", "device": "cpu", "concurrency": 1},
        "unique_tasks": len(tasks), "repeats": repeats, "warmup_passes": warmup,
        "measured_task_runs": len(samples), "successful_task_runs": len(samples), "failed_task_runs": 0,
        "p50_latency_ms": quantile(latencies, 0.5), "p95_latency_ms": quantile(latencies, 0.95),
        "mean_latency_ms": statistics.mean(latencies), "measured_wall_seconds": wall_seconds,
        "successful_tasks_per_second": len(samples) / wall_seconds if wall_seconds else None,
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dataset-kind", choices=("unverified", "synthetic_demo", "human_test_set"), default="unverified")
    args = parser.parse_args(argv)
    try:
        inputs = {"tasks": args.tasks, "rules": args.rules, "label_config": args.label_config}
        outputs = [path for path in (args.output, args.benchmark) if path is not None]
        if len({p.resolve() for p in outputs}) != len(outputs) or {p.resolve() for p in outputs} & {p.resolve() for p in inputs.values()}:
            raise ValueError("Output paths must be distinct and must not overwrite inputs")
        payload = json.loads(args.rules.read_text(encoding="utf-8-sig"))
        groups = read_label_groups(args.label_config)
        compile_start = time.perf_counter_ns()
        rules = compile_rules(payload, groups)
        compile_ms = (time.perf_counter_ns() - compile_start) / 1_000_000
        predictions, benchmark = run_baseline(json.loads(args.tasks.read_text(encoding="utf-8-sig")), groups, rules, args.repeats, args.warmup)
        benchmark.update({"rules_version": payload["version"], "rules_status": payload.get("status", "unverified"), "compile_ms": compile_ms, "dataset_kind": args.dataset_kind, "measured_at_utc": datetime.now(timezone.utc).isoformat()})
        benchmark["inputs"] = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in inputs.items()}
        benchmark["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.benchmark is not None:
            args.benchmark.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, re.error, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"Regex predictions={len(predictions)}; rules version={payload['version']}; dataset={args.dataset_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
