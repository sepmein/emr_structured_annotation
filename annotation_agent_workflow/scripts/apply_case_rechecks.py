#!/usr/bin/env python3
"""Apply a small audited recheck set to an annotation-results JSON file."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("recheck", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    recheck = json.loads(args.recheck.read_text(encoding="utf-8"))
    replacements = {task["task_id"]: task for task in recheck["tasks"]}
    seen = set()
    result = deepcopy(base)
    for task in result["tasks"]:
        task_id = task["task_id"]
        if task_id not in replacements:
            continue
        update = replacements[task_id]
        for field in ("case_decision", "evidence", "issue_refs"):
            if field in update:
                task[field] = update[field]
        seen.add(task_id)
    missing = set(replacements) - seen
    if missing:
        raise ValueError(f"recheck task IDs missing from base: {sorted(missing)}")
    result["post_recheck"] = {
        "source": str(args.recheck),
        "updated_tasks": sorted(seen),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
