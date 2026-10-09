"""Split reviewed tasks by connected patient and exact-text groups, without a model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any
import xml.etree.ElementTree as ET

if __package__:
    from .evaluate_entity_predictions import normalize_records, read_label_groups
else:
    from evaluate_entity_predictions import normalize_records, read_label_groups


SPLITS = ("train", "validation", "test")


def split_reference(
    payload: Any, label_groups: dict[str, list[str]], ratios: tuple[float, float, float], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
        raise ValueError("Three finite positive split ratios must sum to 1")
    labels = [label for group in label_groups.values() for label in group]
    normalized = normalize_records(payload, set(labels), reference=True, source="reference")
    rows = {str(row["task_id"]): row for row in payload}
    patient_ids = {}
    for task_id, row in rows.items():
        patient = row.get("patient_id")
        if isinstance(patient, bool) or not isinstance(patient, (str, int)) or not str(patient).strip():
            raise ValueError("Every task requires a stable nonempty patient_id before splitting")
        patient_ids[task_id] = str(patient)
    parents = {task_id: task_id for task_id in normalized}

    def find(task_id: str) -> str:
        while parents[task_id] != task_id:
            parents[task_id] = parents[parents[task_id]]
            task_id = parents[task_id]
        return task_id

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    by_patient, by_text = {}, {}
    for task_id in sorted(normalized):
        patient, text = patient_ids[task_id], normalized[task_id].text
        if patient in by_patient:
            union(task_id, by_patient[patient])
        else:
            by_patient[patient] = task_id
        if text in by_text:
            union(task_id, by_text[text])
        else:
            by_text[text] = task_id
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(normalized):
        grouped[find(task_id)].append(task_id)
    components = sorted(grouped.values())
    if len(components) < 3:
        raise ValueError("Fewer than three independent patient/exact-text groups; cannot create three nonempty splits")
    random.Random(seed).shuffle(components)
    components.sort(key=len, reverse=True)
    counts = {name: 0 for name in SPLITS}
    target = {name: len(rows) * ratio for name, ratio in zip(SPLITS, ratios)}
    assignment = {}
    group_for_task = {}
    group_details = []
    for index, component in enumerate(components):
        empty = [name for name in SPLITS if counts[name] == 0]
        candidates = empty if len(components) - index == len(empty) else list(SPLITS)
        size = len(component)

        def cost(name: str) -> float:
            # Greedy task-count balancing. Whole connected groups always stay intact.
            return ((counts[name] + size - target[name]) ** 2 - (counts[name] - target[name]) ** 2) / target[name]

        chosen = min(candidates, key=cost)
        counts[chosen] += size
        group_id = hashlib.sha256(json.dumps(component, ensure_ascii=False).encode("utf-8")).hexdigest()
        group_details.append({
            "group_id": group_id, "split": chosen, "task_count": size,
            "patient_count": len({patient_ids[t] for t in component}),
            "unique_text_count": len({normalized[t].text for t in component}),
        })
        for task_id in component:
            assignment[task_id] = chosen
            group_for_task[task_id] = group_id
    outputs = {name: [rows[t] for t in sorted(rows) if assignment[t] == name] for name in SPLITS}
    patient_sets = {name: {patient_ids[str(row["task_id"])] for row in split} for name, split in outputs.items()}
    text_sets = {name: {row["text"] for row in split} for name, split in outputs.items()}
    for index, name in enumerate(SPLITS):
        for other in SPLITS[index + 1:]:
            if patient_sets[name] & patient_sets[other] or text_sets[name] & text_sets[other]:
                raise ValueError("Internal grouping invariant failed; no split written")
    summaries, entity_support, positive_tasks, class_counts = {}, {}, {}, {}
    for name, split in outputs.items():
        entity_support[name] = Counter(e["label"] for row in split for e in row["entities"])
        positive_tasks[name] = Counter(label for row in split for label in {e["label"] for e in row["entities"]})
        class_counts[name] = Counter()
        missing_case = 0
        for row in split:
            case_choices = row.get("case_choices", {})
            if not isinstance(case_choices, dict):
                raise ValueError("case_choices must be an object when present")
            decision = case_choices.get("case_decision")
            if decision is None:
                missing_case += 1
            elif not isinstance(decision, str) or not decision.strip():
                raise ValueError("case_decision must be a nonempty string when present")
            else:
                class_counts[name][decision] += 1
        summaries[name] = {
            "tasks": len(split), "patients": len(patient_sets[name]), "unique_texts": len(text_sets[name]),
            "target_task_count": target[name], "actual_task_fraction": len(split) / len(rows),
            "reference_entities": sum(entity_support[name].values()),
            "no_entity_tasks": sum(not row["entities"] for row in split),
            "case_decision_counts": dict(class_counts[name]), "missing_case_decision_tasks": missing_case,
        }
    coverage = {
        label: {
            "entities": {name: entity_support[name][label] for name in SPLITS},
            "positive_tasks": {name: positive_tasks[name][label] for name in SPLITS},
        }
        for label in labels
    }
    observed = {label for label in labels if any(entity_support[name][label] for name in SPLITS)}
    warnings = []
    for name in SPLITS:
        unsupported = [label for label in labels if label in observed and entity_support[name][label] == 0]
        if unsupported:
            warnings.append({"split": name, "observed_labels_without_examples": unsupported})
    manifest = {
        "protocol_version": 1, "seed": seed,
        "grouping": "connected_components_of_patient_id_and_identical_original_text",
        "requested_ratios": dict(zip(SPLITS, ratios)),
        "tasks": len(rows), "patients": len(set(patient_ids.values())),
        "unique_texts": len(by_text), "independent_groups": len(components),
        "verified_no_patient_overlap": True, "verified_no_exact_text_overlap": True,
        "near_duplicate_check": "not_performed",
        "splits": summaries, "label_coverage": coverage,
        "labels_absent_from_entire_batch": [label for label in labels if label not in observed],
        "coverage_warnings": warnings,
        "groups": sorted(group_details, key=lambda row: row["group_id"]),
        "assignments": [
            {"task_id": task_id, "split": assignment[task_id], "group_id": group_for_task[task_id],
             "text_sha256": hashlib.sha256(normalized[task_id].text.encode("utf-8")).hexdigest()}
            for task_id in sorted(rows)
        ],
        "notes": [
            "Group sizes can prevent exact requested ratios; all actual counts are reported.",
            "This is seeded group allocation with task-count balancing, not guaranteed label stratification.",
            "No records are dropped, and reviewed no-entity records are retained.",
            "Hashes are provenance aids, not deidentification certification.",
            "Group separation does not prove adjudication, model/test independence or cross-site generalization.",
            "Do not replace or retune a final test set after observing its model scores.",
        ],
    }
    return outputs, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs=3, required=True, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.reference.read_bytes()
        groups = read_label_groups(args.label_config)
        splits, manifest = split_reference(json.loads(raw.decode("utf-8-sig")), groups, tuple(args.ratios), args.seed)
        manifest["source_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest["label_config_sha256"] = hashlib.sha256(args.label_config.read_bytes()).hexdigest()
        outputs = {f"{name}.json": rows for name, rows in splits.items()}
        outputs["split_manifest.json"] = manifest
        for name in outputs:
            if (args.output_dir / name).exists():
                raise ValueError("Use a fresh output directory; existing split artifacts are never overwritten")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print("Grouped task split: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
    print(f"Independent groups={manifest['independent_groups']}; label coverage warnings={len(manifest['coverage_warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
