#!/usr/bin/env python3
"""Compare case-level decisions from two annotation runs and a reference.

The script intentionally evaluates only the four-way case decision. Entity and
relation agreement require stable span offsets, which the pilot output contract
does not yet require.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("目标病例信号", "待专业复核", "非目标", "信息不足")


def load_decisions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ValueError(f"{path}: expected an object with tasks[]")
    result: dict[str, str] = {}
    for index, task in enumerate(data["tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"{path}: tasks[{index}] is not an object")
        task_id = task.get("task_id")
        decision = task.get("case_decision")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}: tasks[{index}] lacks task_id")
        if decision not in CATEGORIES:
            raise ValueError(f"{path}: {task_id} has invalid case_decision {decision!r}")
        if task_id in result:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        result[task_id] = decision
    return result


def kappa(a: list[str], b: list[str]) -> tuple[float, float, float]:
    if not a or len(a) != len(b):
        raise ValueError("decision lists must be non-empty and equal length")
    total = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / total
    a_counts, b_counts = Counter(a), Counter(b)
    expected = sum((a_counts[c] / total) * (b_counts[c] / total) for c in CATEGORIES)
    score = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
    return observed, expected, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="task ID or comma-separated task IDs to exclude; may be repeated",
    )
    args = parser.parse_args()

    run_a, run_b = load_decisions(args.run_a), load_decisions(args.run_b)
    excluded = {
        task_id.strip()
        for value in args.exclude
        for task_id in value.split(",")
        if task_id.strip()
    }
    common = sorted((set(run_a) & set(run_b)) - excluded)
    missing = {
        "only_in_a": sorted(set(run_a) - set(run_b)),
        "only_in_b": sorted(set(run_b) - set(run_a)),
    }
    observed, expected, score = kappa([run_a[x] for x in common], [run_b[x] for x in common])
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements = []
    for task_id in common:
        confusion[run_a[task_id]][run_b[task_id]] += 1
        if run_a[task_id] != run_b[task_id]:
            disagreements.append({"task_id": task_id, "run_a": run_a[task_id], "run_b": run_b[task_id]})

    report = {
        "common_tasks": len(common),
        "excluded_tasks": sorted(excluded),
        "observed_agreement": round(observed, 6),
        "expected_agreement": round(expected, 6),
        "cohens_kappa": round(score, 6),
        "missing": missing,
        "category_counts": {
            "run_a": Counter(run_a[task_id] for task_id in common),
            "run_b": Counter(run_b[task_id] for task_id in common),
        },
        "confusion_run_a_by_run_b": {key: dict(value) for key, value in confusion.items()},
        "disagreements": disagreements,
    }

    if args.reference:
        reference = load_decisions(args.reference)
        for name, run in (("run_a", run_a), ("run_b", run_b)):
            shared = sorted((set(run) & set(reference)) - excluded)
            report[f"{name}_vs_reference"] = {
                "tasks": len(shared),
                "agreement": round(sum(run[x] == reference[x] for x in shared) / len(shared), 6),
                "disagreements": [
                    {"task_id": x, "run": run[x], "reference": reference[x]}
                    for x in shared
                    if run[x] != reference[x]
                ],
            }

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
