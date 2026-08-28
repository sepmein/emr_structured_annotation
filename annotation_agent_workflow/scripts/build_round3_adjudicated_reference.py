#!/usr/bin/env python3
"""Build the Round 3 case-level adjudicated guide-development reference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistered", type=Path)
    parser.add_argument("recommendations", type=Path)
    parser.add_argument("duplicate_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    base = json.loads(args.preregistered.read_text(encoding="utf-8"))
    review = json.loads(args.recommendations.read_text(encoding="utf-8"))
    duplicates = json.loads(args.duplicate_audit.read_text(encoding="utf-8"))
    adjudications = {x["task_id"]: x for x in review["case_adjudications"]}

    tasks = []
    changes_from_reference = []
    for task in base["tasks"]:
        item = {
            "task_id": task["task_id"],
            "case_decision": task["case_decision"],
            "evidence": task.get("evidence", []),
            "data_issue": task.get("data_issue", ["none"]),
        }
        if task["task_id"] in adjudications:
            adjudication = adjudications[task["task_id"]]
            before = item["case_decision"]
            item["case_decision"] = adjudication["recommended_case_decision"]
            item["evidence"] = adjudication["minimal_deidentified_evidence"]
            item["adjudication_basis"] = adjudication["reason"]
            if before != item["case_decision"]:
                changes_from_reference.append(
                    {"task_id": item["task_id"], "before": before, "after": item["case_decision"]}
                )
        tasks.append(item)

    excluded = set(duplicates["excluded_duplicate_tasks"])
    unique_tasks = [task for task in tasks if task["task_id"] not in excluded]
    output = {
        "reference_id": "round-03-adjudicated-guide-development-reference-v2.1.0",
        "source_guide_version": "2.0.1",
        "target_guide_version": "2.1.0",
        "purpose": "200例AI指南压力测试的病例级裁决参考；不是临床诊断、真人IAA或临床金标准",
        "raw_distribution": dict(Counter(x["case_decision"] for x in tasks)),
        "unique_text_distribution": dict(Counter(x["case_decision"] for x in unique_tasks)),
        "raw_tasks": len(tasks),
        "unique_exact_text_tasks": len(unique_tasks),
        "changes_from_preregistered_reference": changes_from_reference,
        "adjudicated_task_ids": sorted(adjudications),
        "duplicate_exclusion_rule": duplicates["primary_analysis_rule"],
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("raw_distribution", "unique_text_distribution")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
