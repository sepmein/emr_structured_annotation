#!/usr/bin/env python3
"""Build a deterministic, stratified 100-case de-identified Round 2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from build_deidentified_pilot import CASE_INDEXES as ROUND1_INDEXES
from build_deidentified_pilot import KEYWORDS, redact


SEED = 20260828
TARGETS = {
    "direct": 21,
    "acute_resp": 30,
    "pathogen": 6,
    "chest": 11,
    "chronic_resp": 2,
    "other": 20,
    "low_info": 5,
    "blank": 5,
}

DIRECT = re.compile(
    r"支气管肺炎|大叶性肺炎|间质性肺炎|肺部感染|两肺感染|双肺感染|"
    r"[左右双两]?肺.{0,4}(?:炎症|实变|感染性病变)|"
    r"以[“\"]?肺炎[”\"]?(?:收治|收入|入院)|诊断.{0,5}肺炎"
)
FEVER = re.compile(r"发热|高热|低热|体温.{0,5}(?:3[789]|4[01])")
RESPIRATORY = re.compile(r"咳嗽|咳痰|气促|呼吸困难|低氧|喘息|啰音|呼吸音粗")
PATHOGEN = re.compile(
    r"肺炎支原体|肺炎衣原体|呼吸道合胞病毒|新型冠状病毒|流感病毒|"
    r"肺炎克雷伯菌|军团菌|腺病毒|曲霉|隐球菌"
)
TEST_RESULT = re.compile(r"阳性|阴性|检出|培养见|测序提示|PCR|核酸")
CHEST = re.compile(r"胸水|胸腔积液|脓胸|胸膜|斑片影|条索影|结节|肿块|肺不张")
CHRONIC = re.compile(r"反复.{0,30}(?:年|月)|慢性|陈旧性")


def stratum(text: str) -> str:
    stripped = text.strip()
    if not re.sub(r"[\s\W_]+", "", stripped):
        return "blank"
    if len(stripped) < 40:
        return "low_info"
    if DIRECT.search(stripped):
        return "direct"
    if FEVER.search(stripped) and RESPIRATORY.search(stripped):
        return "acute_resp"
    if PATHOGEN.search(stripped) and TEST_RESULT.search(stripped):
        return "pathogen"
    if CHEST.search(stripped):
        return "chest"
    if CHRONIC.search(stripped) and RESPIRATORY.search(stripped):
        return "chronic_resp"
    return "other"


def excerpt_round2(text: str, limit: int = 1600) -> str:
    """Keep narrative plus relevant laboratory lines for the broader pilot."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= limit:
        return text
    prefix = text[:700].rstrip()
    tail = text[700:]
    pieces = re.split(r"(?<=[。；;！？!?])|\n", tail)
    selected = [piece.strip() for piece in pieces if KEYWORDS.search(piece)]
    combined = prefix + "\n【相关后文节选】\n" + "".join(selected)
    return combined[:limit].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    excluded = set(ROUND1_INDEXES)
    pools: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if index in excluded:
            continue
        raw = record.get("data", {}).get("text") or ""
        pools[stratum(raw)].append(index)

    rng = random.Random(SEED)
    selected: list[tuple[int, str]] = []
    for name, target in TARGETS.items():
        candidates = pools[name]
        if len(candidates) < target:
            raise ValueError(f"stratum {name} has {len(candidates)} cases, need {target}")
        chosen = candidates if len(candidates) == target else rng.sample(candidates, target)
        selected.extend((index, name) for index in chosen)
    rng.shuffle(selected)

    tasks = []
    manifest = {
        "round": "round-02",
        "seed": SEED,
        "excluded_round1_source_indexes": sorted(excluded),
        "targets": TARGETS,
        "tasks": [],
    }
    for number, (index, name) in enumerate(selected, start=1):
        record = records[index]
        raw = record.get("data", {}).get("text") or ""
        cleaned = excerpt_round2(redact(raw, record), limit=1600)
        task_id = f"R2-{number:03d}"
        tasks.append(
            {
                "task_id": task_id,
                "source": "真实EMR分层固定抽样去标识化节选",
                "text": cleaned,
            }
        )
        manifest["tasks"].append(
            {
                "task_id": task_id,
                "source_index": index,
                "sampling_stratum": name,
                "deidentified_text_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
