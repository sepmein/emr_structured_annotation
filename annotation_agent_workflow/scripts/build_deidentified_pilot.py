#!/usr/bin/env python3
"""Build a fixed, de-identified EMR excerpt set for annotation-guide testing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CASE_INDEXES = [229, 211, 109, 91, 348, 225, 105, 147, 146, 38, 331, 294, 178, 344, 334, 101, 200, 282, 327, 55]
KEYWORDS = re.compile(
    r"肺炎|肺部感染|肺炎症|发热|咳嗽|咳痰|气促|呼吸|低氧|胸水|胸腔|病原|核酸|抗原|IgM|培养|"
    r"现仍|目前|既往|拟|计划|\d+(?:余|多|\+)?(?:小时|天|周|月|年)|今(?:日|晨)?|次日|加重|好转|无好转|未愈|反复"
)

COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹"
    "苏潘葛范彭鲁韦马方任袁柳史唐薛雷贺倪汤殷罗毕郝安常傅齐康伍余"
    "顾孟黄萧尹姚邵汪毛戴宋熊纪舒屈项董梁杜阮蓝季贾江童颜郭梅林钟"
    "徐邱高夏蔡田樊胡霍万卢莫房裘陆荣翁羊甄家封靳段富焦侯全秋仲宫"
    "宁栾祖武符刘景詹龙叶黎白蒲易廖乔谭温庄柴瞿阎连艾向古慎曾关查"
)


def names_from(value) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"patient_name", "patientname"} and isinstance(child, str):
                text = child.strip()
                if 2 <= len(text) <= 12 and "*" not in text:
                    names.add(text)
            names.update(names_from(child))
    elif isinstance(value, list):
        for child in value:
            names.update(names_from(child))
    return names


def redact(text: str, record: dict) -> str:
    for name in sorted(names_from(record), key=len, reverse=True):
        text = text.replace(name, "[姓名已删除]")
    text = re.sub(r"患者[：:]\s*[\u4e00-\u9fff·]{2,8}", "患者[姓名已删除]", text)

    def redact_patient_prefix(match: re.Match[str]) -> str:
        candidate = match.group("candidate")
        if len(candidate) in {2, 3} and candidate[0] in COMMON_SURNAMES:
            return "患者[姓名已删除]"
        return match.group(0)

    text = re.sub(
        r"患者(?P<candidate>[\u4e00-\u9fff]{2,4})(?=[，,])",
        redact_patient_prefix,
        text,
    )
    text = re.sub(r"患者[\u4e00-\u9fff]{2,4}(?=[，,](?:男|女))", "患者[姓名已删除]", text)
    text = re.sub(
        r"(?:患者姓名|姓名|住院编号|住院号|床位号|床位|床号)[：:]?\s*[^，,。；;\n]{0,40}",
        "[身份字段已删除]",
        text,
    )
    text = re.sub(r"(?<![\d.-])\d{7,18}(?![\d.-])", "[长编号已删除]", text)
    text = re.sub(r"(?:住址|地址)[：:]?[^\n。；;]{2,80}", "地址：[已删除]", text)
    text = re.sub(r"1[3-9]\d{9}", "[电话已删除]", text)
    return text


def excerpt(text: str, limit: int = 1100) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    narrative = re.split(r"【检验项目】|【检验数值】", text, maxsplit=1)[0]
    if len(narrative) <= limit:
        return narrative.strip()
    prefix = narrative[:650]
    tail = narrative[650:]
    pieces = re.split(r"(?<=[。；;！？!?])|\n", tail)
    selected = [piece.strip() for piece in pieces if KEYWORDS.search(piece)]
    combined = prefix.rstrip() + "\n【相关后文节选】\n" + "".join(selected)
    return combined[:limit].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    output = []
    for number, index in enumerate(CASE_INDEXES, start=1):
        record = records[index]
        raw = record.get("data", {}).get("text") or ""
        cleaned = excerpt(redact(raw, record))
        output.append(
            {
                "task_id": f"R1-{number:03d}",
                "source": "真实EMR固定索引去标识化节选",
                "text": cleaned,
            }
        )
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
