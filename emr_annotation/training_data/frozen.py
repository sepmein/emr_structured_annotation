"""Validate reviewed patient splits, preserving negative records for NER+RE.

Standard library only. Held-out data are checked for separation and never
returned as training/validation data. This is not medical or tokenizer review.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from emr_annotation.evaluation.entity_predictions import normalize_records, read_label_groups


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("A stable nonempty identifier is required")
    return str(value)


def convert_rows(rows, labels, relation_labels):
    normalize_records(rows, labels, reference=True, source="training conversion")
    converted = []
    for row in rows:
        entities, ids = [], set()
        for index, original in enumerate(row["entities"]):
            entity_id = identifier(original.get("id", f"generated_entity_{index}"))
            if entity_id in ids:
                raise ValueError("Entity IDs must be unique for relation training")
            ids.add(entity_id)
            entities.append({"id": entity_id, "start": original["start"], "end": original["end"], "label": original["label"]})
        ordered = sorted(entities, key=lambda e: (e["start"], e["end"]))
        for previous, current in zip(ordered, ordered[1:]):
            if current["start"] < previous["end"]:
                raise ValueError("Overlapping entities require a different single-BIO training policy; review task " + identifier(row["task_id"]))
        raw_relations = row.get("relations", [])
        if not isinstance(raw_relations, list):
            raise ValueError("relations must be an array")
        relations, pair_types = [], {}
        for relation in raw_relations:
            if not isinstance(relation, dict):
                raise ValueError("Relation must be an object")
            source, target = identifier(relation.get("from_id")), identifier(relation.get("to_id"))
            kind, direction = relation.get("type"), relation.get("direction", "right")
            if source not in ids or target not in ids or source == target:
                raise ValueError("Relation requires two distinct existing entity IDs")
            if not isinstance(kind, str) or kind not in relation_labels or direction not in ("right", "left", "bi"):
                raise ValueError("Relation type/direction must match the schema contract")
            pairs = [(source, target)] if direction == "right" else [(target, source)] if direction == "left" else [(source, target), (target, source)]
            for pair in pairs:
                if pair in pair_types:
                    raise ValueError("Duplicate/multiple relation types for one directed pair are unsupported")
                pair_types[pair] = kind
                relations.append({"from_id": pair[0], "to_id": pair[1], "type": kind})
        converted.append({"task_id": identifier(row["task_id"]), "text": row["text"], "entities": entities, "relations": relations})
    return converted


def prepare_frozen_training(splits_dir: Path, reference_path: Path, label_config: Path):
    groups = read_label_groups(label_config)
    ner_labels = [label for group in groups.values() for label in group]
    re_labels = []
    for node in ET.parse(label_config).getroot().iter():
        if node.tag.rsplit("}", 1)[-1] == "Relation" and node.get("value") and node.get("value") not in re_labels:
            re_labels.append(node.get("value"))
    if "无关系" in re_labels:
        raise ValueError("Relations must not reuse the reserved no-relation label")
    paths = {name: splits_dir / f"{name}.json" for name in ("train", "validation", "test")}
    paths.update({"reference": reference_path, "label_config": label_config, "manifest": splits_dir / "split_manifest.json"})
    raw = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    manifest = json.loads(raw["manifest"].decode("utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("source_sha256") != hashes["reference"] or manifest.get("label_config_sha256") != hashes["label_config"]:
        raise ValueError("Manifest does not match the reviewed source or frozen XML")
    source = json.loads(raw["reference"].decode("utf-8-sig"))
    normalize_records(source, set(ner_labels), reference=True, source="reviewed reference")
    source_by_id = {identifier(row["task_id"]): row for row in source}
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Manifest requires explicit task assignments")
    assigned = {}
    for item in assignments:
        if not isinstance(item, dict):
            raise ValueError("Invalid task assignment")
        task_id = identifier(item.get("task_id"))
        if task_id in assigned or item.get("split") not in ("train", "validation", "test"):
            raise ValueError("Duplicate assignment or unknown split")
        assigned[task_id] = item
    if set(assigned) != set(source_by_id):
        raise ValueError("Manifest and reviewed source task IDs differ")
    splits, patients, texts, seen, summaries = {}, {}, {}, set(), {}
    for name in ("train", "validation", "test"):
        rows = json.loads(raw[name].decode("utf-8-sig"))
        normalize_records(rows, set(ner_labels), reference=True, source=name)
        splits[name] = rows
        patients[name], texts[name] = set(), set()
        for row in rows:
            task_id = identifier(row["task_id"])
            if task_id in seen or task_id not in source_by_id or row != source_by_id[task_id]:
                raise ValueError("Split task repeats, is unknown, or differs from reviewed source")
            seen.add(task_id)
            if assigned[task_id]["split"] != name or assigned[task_id].get("text_sha256") != hashlib.sha256(row["text"].encode("utf-8")).hexdigest():
                raise ValueError("Task does not match its frozen assignment")
            patients[name].add(identifier(row.get("patient_id")))
            texts[name].add(row["text"])
        support = Counter(e["label"] for row in rows for e in row["entities"])
        summaries[name] = {"tasks": len(rows), "patients": len(patients[name]), "unique_texts": len(texts[name]),
                           "no_entity_tasks": sum(not row["entities"] for row in rows), "reference_entities": sum(support.values()),
                           "entity_support": {label: support[label] for label in ner_labels}}
    if seen != set(source_by_id):
        raise ValueError("Some reviewed tasks are missing from the splits")
    for index, name in enumerate(("train", "validation", "test")):
        for other in ("train", "validation", "test")[index + 1:]:
            if patients[name] & patients[other] or texts[name] & texts[other]:
                raise ValueError("Patient or exact-text overlap across splits")
    train = convert_rows(splits["train"], set(ner_labels), re_labels)
    validation = convert_rows(splits["validation"], set(ner_labels), re_labels)
    schema_json = json.dumps({"ner_labels": ner_labels, "re_labels": re_labels}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    receipt = {
        "protocol_version": 1, "status": "data_preflight_passed_training_not_run",
        "schema_id": "schema_" + hashlib.sha256(schema_json.encode("utf-8")).hexdigest()[:12],
        "ner_labels": ner_labels, "re_labels": re_labels,
        "inputs": {name: {"file": paths[name].name, "sha256": value} for name, value in hashes.items()}, "splits": summaries,
        "verified_no_patient_overlap": True, "verified_no_exact_text_overlap": True,
        "training_policy": {"resplit": False, "oversampling": False, "keep_no_entity_tasks": True,
                            "test_passed_to_training": False, "relation_left_swapped": True, "relation_bi_expanded": True},
        "notes": [
            "Matching hashes/rows do not certify medical adjudication or unseen test status.",
            "No near-duplicate check is performed.",
            "Character overlaps are rejected; tokenizer collisions, truncation and alignment need runtime review.",
            "Attributes/case choices remain in original references but are not learned by NER+RE.",
            "Training selection uses legacy token macro NER and gold-entity-pair RE, not strict final entity scores.",
        ],
    }
    return train, validation, receipt
