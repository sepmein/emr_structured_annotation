#!/usr/bin/env python3
"""Validate the handoff from the medical annotator to the guide developer."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CATEGORIES = {
    "definition_missing",
    "boundary",
    "overlap",
    "relation",
    "decision_relevance",
    "xml_guide_mismatch",
    "data_quality",
    "workflow",
}
SEVERITIES = {"blocker", "major", "minor"}
REQUIRED_ISSUE_FIELDS = {
    "issue_id",
    "category",
    "severity",
    "affected_tasks",
    "affected_labels",
    "evidence_excerpt",
    "observed_problem",
    "question",
    "decision_impact",
    "temporary_action",
    "suggested_options",
}
FORBIDDEN_KEYS = {"patient_id", "patient_name", "id_card", "id_card_no", "phone", "address"}


def fail(message: str) -> None:
    raise ValueError(message)


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail("top level must be an object")
    for field in ("run_id", "guide_version", "xml_sha256", "sample", "issues"):
        if field not in data:
            fail(f"missing top-level field: {field}")
    sample = data["sample"]
    if not isinstance(sample, dict) or not {"total", "completed", "blocked"} <= sample.keys():
        fail("sample must contain total, completed, and blocked")
    if any(not isinstance(sample[k], int) or sample[k] < 0 for k in ("total", "completed", "blocked")):
        fail("sample counts must be non-negative integers")
    if sample["completed"] + sample["blocked"] > sample["total"]:
        fail("completed + blocked cannot exceed total")
    if not isinstance(data["issues"], list):
        fail("issues must be an array")
    seen = set()
    for index, issue in enumerate(data["issues"]):
        if not isinstance(issue, dict):
            fail(f"issues[{index}] must be an object")
        missing = REQUIRED_ISSUE_FIELDS - issue.keys()
        if missing:
            fail(f"issues[{index}] missing fields: {sorted(missing)}")
        issue_id = issue["issue_id"]
        if not isinstance(issue_id, str) or not issue_id:
            fail(f"issues[{index}].issue_id must be a non-empty string")
        if issue_id in seen:
            fail(f"duplicate issue_id: {issue_id}")
        seen.add(issue_id)
        if issue["category"] not in CATEGORIES:
            fail(f"{issue_id}: invalid category")
        if issue["severity"] not in SEVERITIES:
            fail(f"{issue_id}: invalid severity")
        if not isinstance(issue["affected_tasks"], list) or not isinstance(issue["affected_labels"], list):
            fail(f"{issue_id}: affected_tasks and affected_labels must be arrays")
        if not isinstance(issue["suggested_options"], list):
            fail(f"{issue_id}: suggested_options must be an array")
    leaked = FORBIDDEN_KEYS.intersection(walk_keys(data))
    if leaked:
        fail(f"forbidden identifying keys present: {sorted(leaked)}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_feedback.py annotator_feedback.json", file=sys.stderr)
        return 2
    try:
        validate(Path(sys.argv[1]))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
