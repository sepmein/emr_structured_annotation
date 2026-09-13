#!/usr/bin/env python3
"""Validate coverage and card fields for the v2.0.1 detailed label dictionary."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = ROOT / "guides" / "label_dictionary_v2.1.0.md"
XML_FILES = [
    Path("/Users/sepmein/x/projects/emr_structured_annotation/label_studio/pnemonia_adult_config.xml"),
    Path("/Users/sepmein/x/projects/emr_structured_annotation/label_studio/pneumonia_children_config.xml"),
]

ALIASES: dict[str, str] = {}

EXTERNAL_CARDS = {
    "目标病例信号",
    "待专业复核",
    "非目标",
    "信息不足",
    "咳嗽",
    "咳痰",
    "气促/呼吸困难",
    "TimeExpression",
    "task_id",
    "evidence",
    "annotation_status：completed",
    "annotation_status：blocked_privacy",
    "data_issue：none",
    "data_issue：empty_text",
    "data_issue：missing_context",
    "data_issue：truncated",
    "data_issue：duplicate_text",
    "data_issue：record_splice",
    "data_issue：lab_pairing_error",
    "data_issue：template_conflict",
    "data_issue：privacy_block",
    "data_issue：other",
    "data_issue_note",
}

REQUIRED_FIELDS = [
    "医学定义",
    "项目操作定义",
    "纳入标准",
    "排除标准",
    "正例1",
    "正例2",
    "近似反例",
    "常见混淆与裁决",
]


def xml_schema_items() -> tuple[set[str], set[str]]:
    labels: set[str] = set()
    relations: set[str] = set()
    for path in XML_FILES:
        root = ET.parse(path).getroot()
        labels.update(node.attrib["value"] for node in root.iter("Label"))
        relations.update(node.attrib["value"] for node in root.iter("Relation"))
    return labels, relations


def cards(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^### \d+\.\d+ (.+)$", text))
    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[name] = text[match.end() : end]
    return result


def main() -> int:
    text = DICTIONARY.read_text(encoding="utf-8")
    parsed = cards(text)
    labels, relations = xml_schema_items()
    expected = {ALIASES.get(x, x) for x in labels} | relations | EXTERNAL_CARDS

    missing = sorted(expected - parsed.keys())
    problems: list[str] = []
    if missing:
        problems.append("missing cards: " + ", ".join(missing))

    for name in sorted(expected & parsed.keys()):
        body = parsed[name]
        for field in REQUIRED_FIELDS:
            if f"**{field}**" not in body:
                problems.append(f"{name}: missing field {field}")
        if "**允许关系**" not in body and "**允许端点和角色**" not in body and "**跨度/关系**" not in body:
            problems.append(f"{name}: missing relationship/form rule")
        if "**最小完整跨度**" not in body and "**跨度**" not in body and "**跨度/关系**" not in body:
            problems.append(f"{name}: missing span/form rule")
        if "**病例判断作用**" not in body and "**病例作用**" not in body:
            problems.append(f"{name}: missing case-decision role")

    numbered_sections = [int(x) for x in re.findall(r"(?m)^## (\d+)\. ", text)]
    if numbered_sections != list(range(1, 15)):
        problems.append(f"section order is {numbered_sections}, expected 1..14")

    if problems:
        print("FAIL")
        print("\n".join(problems))
        return 1

    print(
        "PASS: "
        f"{len(labels)} XML labels + {len(relations)} XML relations + "
        f"{len(EXTERNAL_CARDS)} external/recommended cards covered; "
        f"{len(expected)} unique required cards validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
