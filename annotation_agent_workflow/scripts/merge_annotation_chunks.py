#!/usr/bin/env python3
"""Merge four independently produced annotation chunks and feedback notes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


XML_SHA = {
    "adult": "b80e3064a159bfcfe285b808eedf706f232b04acb431f369a4f4075fe6ea39b5",
    "child": "68410b7207b971060bf45defa4717327f48f71a0f1391364ab628ec4a9ee9a50",
}


def merge_issues(paths: list[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        issues = data.get("issues", []) if isinstance(data, dict) else data
        for issue in issues:
            issue_id = issue["issue_id"]
            if issue_id not in merged:
                merged[issue_id] = dict(issue)
                merged[issue_id]["affected_tasks"] = list(issue.get("affected_tasks", []))
                merged[issue_id]["affected_labels"] = list(issue.get("affected_labels", []))
                merged[issue_id]["suggested_options"] = list(issue.get("suggested_options", []))
                merged[issue_id]["observed_in_chunks"] = [path.stem]
            else:
                current = merged[issue_id]
                current["affected_tasks"] = sorted(set(current["affected_tasks"]) | set(issue.get("affected_tasks", [])))
                current["affected_labels"] = sorted(set(current["affected_labels"]) | set(issue.get("affected_labels", [])))
                current["suggested_options"] = list(
                    dict.fromkeys(current["suggested_options"] + issue.get("suggested_options", []))
                )
                current["observed_in_chunks"].append(path.stem)
    for issue in merged.values():
        issue["affected_tasks"] = sorted(set(issue["affected_tasks"]))
        issue["affected_labels"] = sorted(set(issue["affected_labels"]))
    return [merged[key] for key in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotator_dir", type=Path)
    parser.add_argument("annotator_id")
    args = parser.parse_args()

    result_paths = sorted(args.annotator_dir.glob("annotation_results_chunk_*.json"))
    issue_paths = sorted(args.annotator_dir.glob("issue_notes_chunk_*.json"))
    if len(result_paths) != 4 or len(issue_paths) != 4:
        raise SystemExit(f"expected four result and issue chunks, got {len(result_paths)} and {len(issue_paths)}")

    tasks = []
    for path in result_paths:
        tasks.extend(json.loads(path.read_text(encoding="utf-8"))["tasks"])
    tasks.sort(key=lambda item: item["task_id"])
    if len(tasks) != 200 or len({x["task_id"] for x in tasks}) != 200:
        raise SystemExit("merged tasks are not 200 unique records")

    results = {
        "run_id": f"round-03-annotator-{args.annotator_id}",
        "guide_version": "2.0.1",
        "input": "../inputs/pilot_cases_200.json",
        "annotation_scope": "200 de-identified EMR tasks; public-health case-finding signals, not clinical diagnoses",
        "tasks": tasks,
    }
    (args.annotator_dir / "annotation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    issues = merge_issues(issue_paths)
    completed = sum(x["annotation_status"] == "completed" for x in tasks)
    blocked = sum(x["annotation_status"] == "blocked_privacy" for x in tasks)
    feedback = {
        "run_id": f"round-03-annotator-{args.annotator_id}",
        "guide_version": "2.0.1",
        "xml_sha256": XML_SHA,
        "sample": {"total": len(tasks), "completed": completed, "blocked": blocked},
        "issues": issues,
    }
    (args.annotator_dir / "annotator_feedback.json").write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    decisions = Counter(x.get("case_decision") for x in tasks if x["annotation_status"] == "completed")
    categories = Counter(x["category"] for x in issues)
    severities = Counter(x["severity"] for x in issues)
    summary = f"""# Round 3 标注员{args.annotator_id.upper()}运行摘要

- 指南版本：v2.0.1
- 样本：200例，完成{completed}例，隐私阻塞{blocked}例
- 病例级分布：目标病例信号{decisions['目标病例信号']}、待专业复核{decisions['待专业复核']}、非目标{decisions['非目标']}、信息不足{decisions['信息不足']}
- 唯一结构化问题：{len(issues)}项；类别分布：{dict(categories)}；严重度：{dict(severities)}
- 本运行输出是指南压力测试结果，不是临床诊断或真人一致性证据。
- 是否建议扩大试标：应先裁决本轮病例级分歧、修复重复/空文本门禁并更新指南，再进入真人校准。
"""
    (args.annotator_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "issues": len(issues), "decisions": dict(decisions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
