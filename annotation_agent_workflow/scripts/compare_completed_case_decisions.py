#!/usr/bin/env python3
"""Compare four-way case decisions while keeping privacy-blocked tasks separate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("目标病例信号", "待专业复核", "非目标", "信息不足")


def load(path: Path) -> tuple[dict[str, str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    decisions: dict[str, str] = {}
    blocked: set[str] = set()
    for task in data["tasks"]:
        task_id = task["task_id"]
        status = task.get("annotation_status", "completed")
        if status == "blocked_privacy":
            blocked.add(task_id)
            continue
        decision = task.get("case_decision")
        if decision not in CATEGORIES:
            raise ValueError(f"{path}: {task_id} invalid decision {decision!r}")
        decisions[task_id] = decision
    return decisions, blocked


def load_reference(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        task["task_id"]: task["case_decision"]
        for task in data["tasks"]
        if task.get("case_decision") in CATEGORIES
    }


def kappa(a: list[str], b: list[str]) -> tuple[float, float, float]:
    total = len(a)
    if total == 0 or total != len(b):
        raise ValueError("no equal-length completed comparison set")
    observed = sum(x == y for x, y in zip(a, b)) / total
    ac, bc = Counter(a), Counter(b)
    expected = sum((ac[c] / total) * (bc[c] / total) for c in CATEGORIES)
    score = 1.0 if observed == expected == 1.0 else (observed - expected) / (1 - expected)
    return observed, expected, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--input", type=Path, help="Original task array, used for exact-text collapse")
    parser.add_argument("--collapse-exact-text", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    a, blocked_a = load(args.run_a)
    b, blocked_b = load(args.run_b)
    excluded = blocked_a | blocked_b
    exact_text_groups: list[list[str]] = []
    duplicate_excluded: set[str] = set()
    if args.collapse_exact_text:
        if not args.input:
            raise ValueError("--collapse-exact-text requires --input")
        input_tasks = json.loads(args.input.read_text(encoding="utf-8"))
        by_text: dict[str, list[str]] = defaultdict(list)
        for task in input_tasks:
            by_text[task.get("text") or ""].append(task["task_id"])
        for task_ids in by_text.values():
            if len(task_ids) > 1:
                exact_text_groups.append(task_ids)
                duplicate_excluded.update(task_ids[1:])
    common = sorted((set(a) & set(b)) - excluded - duplicate_excluded)
    observed, expected, score = kappa([a[x] for x in common], [b[x] for x in common])

    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements = []
    for task_id in common:
        matrix[a[task_id]][b[task_id]] += 1
        if a[task_id] != b[task_id]:
            disagreements.append({"task_id": task_id, "run_a": a[task_id], "run_b": b[task_id]})

    report = {
        "comparison_scope": "completed tasks only; privacy blocks excluded",
        "common_completed_tasks": len(common),
        "blocked": {"run_a": sorted(blocked_a), "run_b": sorted(blocked_b), "union": sorted(excluded)},
        "exact_text_collapse": {
            "enabled": args.collapse_exact_text,
            "groups": exact_text_groups,
            "excluded_duplicate_tasks": sorted(duplicate_excluded),
        },
        "observed_agreement": round(observed, 6),
        "expected_agreement": round(expected, 6),
        "cohens_kappa": round(score, 6),
        "category_counts": {
            "run_a": dict(Counter(a[x] for x in common)),
            "run_b": dict(Counter(b[x] for x in common)),
        },
        "confusion_run_a_by_run_b": {key: dict(value) for key, value in matrix.items()},
        "disagreements": disagreements,
        "entity_relation_note": "Not scored here; strict entity/relation F1 requires stable character offsets and endpoint IDs.",
    }

    if args.reference:
        reference = load_reference(args.reference)
        for name, run in (("run_a", a), ("run_b", b)):
            shared = sorted((set(run) & set(reference)) - excluded - duplicate_excluded)
            report[f"{name}_vs_reference"] = {
                "tasks": len(shared),
                "agreement": round(sum(run[x] == reference[x] for x in shared) / len(shared), 6),
                "disagreements": [
                    {"task_id": x, "run": run[x], "reference": reference[x]}
                    for x in shared
                    if run[x] != reference[x]
                ],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "common_completed_tasks": len(common),
                "blocked_union": len(excluded),
                "observed_agreement": report["observed_agreement"],
                "cohens_kappa": report["cohens_kappa"],
                "disagreements": len(disagreements),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
