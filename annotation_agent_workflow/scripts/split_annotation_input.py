#!/usr/bin/env python3
"""Split an annotation input JSON array into deterministic fixed-size chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tasks), args.size):
        chunk = tasks[start : start + args.size]
        number = start // args.size + 1
        path = args.output_dir / f"pilot_cases_200_chunk_{number:02d}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "chunks": (len(tasks) + args.size - 1) // args.size}))


if __name__ == "__main__":
    main()
