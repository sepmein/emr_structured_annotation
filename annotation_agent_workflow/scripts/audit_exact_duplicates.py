#!/usr/bin/env python3
"""Audit exact duplicate task texts without emitting the text itself."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = defaultdict(list)
    blank_by_hash: dict[str, bool] = {}
    for task in tasks:
        text = task.get("text") or ""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        groups[digest].append(task["task_id"])
        blank_by_hash[digest] = not text.strip()

    duplicate_groups = []
    excluded = []
    for digest, task_ids in groups.items():
        if len(task_ids) < 2:
            continue
        duplicate_groups.append(
            {
                "text_sha256": digest,
                "blank": blank_by_hash[digest],
                "task_ids": task_ids,
                "primary_representative": task_ids[0],
                "excluded_from_unique_text_analysis": task_ids[1:],
            }
        )
        excluded.extend(task_ids[1:])

    report = {
        "input": str(args.input),
        "total_tasks": len(tasks),
        "unique_exact_texts": len(groups),
        "duplicate_groups": duplicate_groups,
        "excluded_duplicate_tasks": excluded,
        "primary_analysis_rule": "Keep first deterministic representative per exact de-identified text; retain all task-level annotations for audit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(tasks), "unique": len(groups), "groups": len(duplicate_groups), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
