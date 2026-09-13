#!/usr/bin/env python3
"""Validate annotation result files against the case-form v2.0.1 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DECISIONS = {"目标病例信号", "待专业复核", "非目标", "信息不足"}
ISSUES = {
    "none",
    "truncated",
    "duplicate_text",
    "record_splice",
    "lab_pairing_error",
    "template_conflict",
    "privacy_block",
    "other",
}


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--input", type=Path, help="Expected task input file")
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        fail("root.tasks must be a list")

    expected_ids = None
    if args.input:
        expected = json.loads(args.input.read_text(encoding="utf-8"))
        expected_ids = [item["task_id"] for item in expected]

    seen = set()
    counts = {"completed": 0, "blocked_privacy": 0}
    for index, task in enumerate(tasks):
        where = f"tasks[{index}]"
        if not isinstance(task, dict):
            fail(f"{where} must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            fail(f"{where}.task_id missing")
        if task_id in seen:
            fail(f"duplicate task_id {task_id}")
        seen.add(task_id)

        status = task.get("annotation_status")
        if status not in counts:
            fail(f"{task_id}: invalid annotation_status {status!r}")
        counts[status] += 1
        data_issue = task.get("data_issue")
        if not isinstance(data_issue, list) or not data_issue:
            fail(f"{task_id}: data_issue must be a non-empty list")
        unknown = set(data_issue) - ISSUES
        if unknown:
            fail(f"{task_id}: unknown data_issue {sorted(unknown)}")
        if "none" in data_issue and len(data_issue) != 1:
            fail(f"{task_id}: none cannot coexist with another data issue")

        evidence = task.get("evidence")
        if not isinstance(evidence, list):
            fail(f"{task_id}: evidence must be a list")
        if status == "blocked_privacy":
            if task.get("case_decision") is not None or evidence:
                fail(f"{task_id}: privacy block requires null decision and empty evidence")
            if "privacy_block" not in data_issue:
                fail(f"{task_id}: privacy block requires privacy_block data issue")
        else:
            decision = task.get("case_decision")
            if decision not in DECISIONS:
                fail(f"{task_id}: invalid case_decision {decision!r}")
            if len(evidence) > 3:
                fail(f"{task_id}: evidence exceeds 3 excerpts")
            if decision != "信息不足" and not evidence:
                fail(f"{task_id}: {decision} requires 1-3 evidence excerpts")

        for field in ("entities", "relations", "issue_refs"):
            if not isinstance(task.get(field), list):
                fail(f"{task_id}: {field} must be a list")

    if expected_ids is not None:
        if set(expected_ids) != seen:
            missing = sorted(set(expected_ids) - seen)
            extra = sorted(seen - set(expected_ids))
            fail(f"task set mismatch; missing={missing}, extra={extra}")
        if len(tasks) != len(expected_ids):
            fail("task count mismatch")

    print(json.dumps({"status": "PASS", "tasks": len(tasks), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
