#!/usr/bin/env python3
"""Write the developer's Round 2 case-level interpretation before agent results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET = {
    "R2-001", "R2-008", "R2-011", "R2-013", "R2-015", "R2-022",
    "R2-023", "R2-024", "R2-031", "R2-044", "R2-045", "R2-060",
    "R2-063", "R2-067", "R2-078", "R2-079", "R2-080", "R2-090",
    "R2-094", "R2-097", "R2-100",
}
REVIEW = {
    "R2-016", "R2-038", "R2-050", "R2-052", "R2-053", "R2-054",
    "R2-055", "R2-095", "R2-099",
}
INSUFFICIENT = {
    "R2-025", "R2-026", "R2-027", "R2-028", "R2-037", "R2-064",
    "R2-085", "R2-086", "R2-091",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.input.read_text(encoding="utf-8"))
    all_ids = {case["task_id"] for case in cases}
    assigned = TARGET | REVIEW | INSUFFICIENT
    unknown = assigned - all_ids
    if unknown:
        raise ValueError(f"assigned IDs not present: {sorted(unknown)}")
    if len(all_ids) != 100:
        raise ValueError(f"expected 100 task IDs, found {len(all_ids)}")

    tasks = []
    for case in cases:
        task_id = case["task_id"]
        if task_id in TARGET:
            decision = "目标病例信号"
        elif task_id in REVIEW:
            decision = "待专业复核"
        elif task_id in INSUFFICIENT:
            decision = "信息不足"
        else:
            decision = "非目标"
        tasks.append({"task_id": task_id, "case_decision": decision})

    output = {
        "reference_id": "round-02-developer-preregistered-v1.0.0",
        "guide_version": "1.0.0",
        "created_before_agent_results": True,
        "purpose": "检验两名试标员如何执行v1.0.0；不是临床金标准，PRE-R2问题可能导致后续裁决改变。",
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
