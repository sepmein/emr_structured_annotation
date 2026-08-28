#!/usr/bin/env python3
"""Build Round 3: 200 new de-identified EMR cases with no Round 1/2 overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from build_deidentified_pilot import CASE_INDEXES as ROUND1_INDEXES
from build_deidentified_pilot import redact
from build_deidentified_pilot_100 import excerpt_round2, stratum


SEED = 20260828
TARGET = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("round2_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    round2 = json.loads(args.round2_manifest.read_text(encoding="utf-8"))
    excluded = set(ROUND1_INDEXES)
    excluded.update(item["source_index"] for item in round2["tasks"])

    remaining: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        if index in excluded:
            continue
        text = record.get("data", {}).get("text") or ""
        remaining.append((index, stratum(text)))

    nonblank = [item for item in remaining if item[1] != "blank"]
    blanks = [item for item in remaining if item[1] == "blank"]
    if len(nonblank) > TARGET:
        raise ValueError(f"nonblank remainder {len(nonblank)} exceeds target {TARGET}")
    blank_needed = TARGET - len(nonblank)
    if len(blanks) < blank_needed:
        raise ValueError(f"only {len(blanks)} blank cases, need {blank_needed}")

    rng = random.Random(SEED)
    selected = nonblank + rng.sample(blanks, blank_needed)
    rng.shuffle(selected)

    tasks = []
    manifest = {
        "round": "round-03",
        "seed": SEED,
        "source_path": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection_rule": "all 173 remaining nonblank cases plus deterministic sample of 27 remaining blank cases",
        "excluded_round1_round2_source_indexes": sorted(excluded),
        "selected_strata": dict(Counter(name for _, name in selected)),
        "tasks": [],
    }

    for number, (index, name) in enumerate(selected, start=1):
        record = records[index]
        raw = record.get("data", {}).get("text") or ""
        cleaned = excerpt_round2(redact(raw, record), limit=1600)
        task_id = f"R3-{number:03d}"
        tasks.append(
            {
                "task_id": task_id,
                "source": "真实EMR未使用病例去标识化节选",
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

    if len(tasks) != TARGET:
        raise AssertionError(len(tasks))
    if len({item["source_index"] for item in manifest["tasks"]}) != TARGET:
        raise AssertionError("duplicate source indexes")
    if excluded & {item["source_index"] for item in manifest["tasks"]}:
        raise AssertionError("Round 1/2 overlap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(tasks), "strata": manifest["selected_strata"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
