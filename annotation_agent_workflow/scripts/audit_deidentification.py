#!/usr/bin/env python3
"""Conservative PHI-pattern audit that reports task IDs only, never matched text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS = {
    "identity_field": re.compile(r"(?:患者姓名|姓名|住院编号|住院号|床位号|床位|床号)[：:]?\s*(?!\[)[^，,。；;\n]{1,40}"),
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "long_identifier": re.compile(r"(?<![\d.-])\d{7,18}(?![\d.-])"),
    "address": re.compile(r"(?:住址|地址)[：:]?\s*(?!\[)[^\n。；;]{2,80}"),
    "patient_name_pattern": re.compile(r"患者[\u4e00-\u9fff·]{2,4}(?=[，,](?:男|女|因|于|以|主诉))"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    findings = []
    for task in tasks:
        text = task.get("text") or ""
        categories = sorted(name for name, pattern in PATTERNS.items() if pattern.search(text))
        if categories:
            findings.append({"task_id": task["task_id"], "categories": categories})

    report = {
        "input": str(args.input),
        "total": len(tasks),
        "flagged": len(findings),
        "category_counts": dict(Counter(x for row in findings for x in row["categories"])),
        "findings": findings,
        "note": "Pattern audit reports task IDs only; flagged records must be reviewed or re-redacted before annotation.",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total", "flagged", "category_counts")}, ensure_ascii=False))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
