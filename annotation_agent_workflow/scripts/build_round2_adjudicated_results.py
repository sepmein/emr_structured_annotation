#!/usr/bin/env python3
"""Build the Round 2 adjudicated case-level reference without overwriting agent runs.

This artifact is a guide-development reference, not a clinical gold standard.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ADJUDICATIONS = {
    "R2-005": {
        "case_decision": "信息不足",
        "basis": "只有泛化的病情稳定/继续治疗，缺少索引疾病、症状和检查，无法确认与任务目标相关。",
        "change_type": "agent_disagreement",
    },
    "R2-031": {
        "case_decision": "目标病例信号",
        "basis": "当前复诊文本明确引用同一连续治疗过程中的两肺炎症及继续治疗，直接证据可归入索引诊疗过程。",
        "change_type": "agent_disagreement",
    },
    "R2-036": {
        "case_decision": "非目标",
        "basis": "完整产科就诊记录足以确认索引事件无关，附带肺炎支原体IgM阴性不构成肺炎信号。",
        "change_type": "agent_disagreement",
    },
    "R2-054": {
        "case_decision": "待专业复核",
        "basis": "已知支气管扩张基础上出现3天内咳嗽加剧和咳痰增多，属于结构性肺病急性呼吸恶化入口。",
        "change_type": "agent_disagreement",
    },
    "R2-074": {
        "case_decision": "信息不足",
        "basis": "腹痛索引记录与后段发热/支原体治疗内容疑似跨记录拼接，归属会反转结论，须退回数据修复。",
        "change_type": "agent_disagreement",
    },
    "R2-078": {
        "case_decision": "目标病例信号",
        "basis": "当前复诊文本明确引用同一连续治疗过程中的两肺炎症及继续治疗，直接证据可归入索引诊疗过程。",
        "change_type": "agent_disagreement",
    },
    "R2-001": {
        "case_decision": "非目标",
        "basis": "非呼吸主诉中偶见少许慢性炎症，急性呼吸症状均被否定，不作为当前感染性肺实质信号。",
        "change_type": "systematic_rule_correction",
    },
    "R2-017": {
        "case_decision": "非目标",
        "basis": "肺占位被影像高度解释为肿瘤并以病理确诊为本次目的，且无急性感染证据。",
        "change_type": "systematic_rule_correction",
    },
    "R2-090": {
        "case_decision": "非目标",
        "basis": "间质性肺炎稳定复诊语境更符合慢性非感染性间质性肺病，未见急性感染或新发肺实质证据。",
        "change_type": "systematic_rule_correction",
    },
    "R2-100": {
        "case_decision": "非目标",
        "basis": "间质性肺炎/呼吸道感染复诊且症状好转、无发热，缺少急性或感染性肺实质证据。",
        "change_type": "systematic_rule_correction",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("output")
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    tasks = base["tasks"]
    by_id = {task["task_id"]: task for task in tasks}
    missing = sorted(set(ADJUDICATIONS) - set(by_id))
    if missing:
        raise SystemExit(f"missing tasks: {missing}")

    final_tasks = []
    changes = []
    for task in tasks:
        # This file is deliberately case-level only. Structured entities from the
        # v1 runs remain in the immutable agent outputs and may be incompatible
        # with v2 boundary changes until a dedicated re-annotation pass.
        updated = {
            "task_id": task["task_id"],
            "case_decision": task["case_decision"],
            "evidence": task.get("evidence", []),
            "issue_refs": task.get("issue_refs", []),
        }
        decision = ADJUDICATIONS.get(task["task_id"])
        if decision:
            before = task["case_decision"]
            updated["case_decision"] = decision["case_decision"]
            updated["adjudication_basis"] = decision["basis"]
            updated["adjudication_change_type"] = decision["change_type"]
            changes.append(
                {
                    "task_id": task["task_id"],
                    "before": before,
                    "after": decision["case_decision"],
                    "change_type": decision["change_type"],
                    "basis": decision["basis"],
                }
            )
        final_tasks.append(updated)

    counts = Counter(task["case_decision"] for task in final_tasks)
    output = {
        "reference_id": "round-02-adjudicated-guide-development-reference",
        "guide_version": "2.0.0",
        "source_run": str(Path(args.base)),
        "purpose": "100例指南压力测试的病例级裁决参考；不是临床诊断或真人金标准",
        "privacy_recheck_applied": ["R2-032", "R2-044", "R2-084"],
        "distribution": dict(counts),
        "adjudications": changes,
        "tasks": final_tasks,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
