"""Frozen adjudicated NER+RE experiment: before -> train -> reload -> after.

Run as a script, not python -m, to preserve local backend imports.
--check-only imports no ML packages and never generates performance numbers.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
from emr_annotation.training_data.token_audit import audit_tokens
from emr_annotation.evaluation.entity_predictions import read_label_groups
from emr_annotation.evaluation.ner_re import joint_report, joint_markdown, relation_keys
from emr_annotation.training_data.frozen import prepare_frozen_training


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix != ".pyc"}


def verify_inputs(args, receipt):
    paths = {name: args.splits_dir / f"{name}.json" for name in ("train", "validation", "test")}
    paths.update(reference=args.reference, label_config=args.label_config,
                 manifest=args.splits_dir / "split_manifest.json")
    if any(sha256(path) != receipt["inputs"][name]["sha256"] for name, path in paths.items()):
        raise ValueError("Frozen inputs changed during experiment; refuse mismatched evaluation")


def label_mappings(receipt, base_model):
    ner = {"O": 0}
    for label in receipt["ner_labels"]:
        ner[f"B-{label}"] = len(ner)
        ner[f"I-{label}"] = len(ner)
    return {"schema_id": receipt["schema_id"], "base_model_name": base_model,
            "ner_label2id": ner, "re_label2id": {"无关系": 0, **{label: i for i, label in enumerate(receipt["re_labels"], 1)}}}


def verify_checkpoint(path, expected):
    actual = json.loads((path / "label_mappings.json").read_text(encoding="utf-8-sig"))
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint base model/schema/label order mismatch; legacy checkpoints need an audited label_mappings.json")
    lora = (path / "adapter_config.json").is_file()
    full = (path / "pytorch_model.bin").is_file()
    if lora == full:
        raise ValueError("Checkpoint must contain exactly one of LoRA or full model weights")
    if lora and (not (path / "classification_heads.pt").is_file() or
                 not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))):
        raise ValueError("Incomplete LoRA checkpoint")
    return lora


def run_experiment(args, train, validation, test, receipt):
    verify_inputs(args, receipt)
    started = time.perf_counter()
    # ML imports intentionally occur only after stdlib preflight.
    import torch
    import transformers
    import peft
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from config import config
    from modeling.network import ModernBERTForNERRE
    from train_ner_re import NERREDataset, collate_fn, compute_label_weights, train_epoch, _save_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config.update_model_paths(args.base_model, None)
    if not Path(config.model_path).is_dir():
        raise ValueError("Local base-model weights missing; populate pretrained_models/<base-model> first")
    smoke_path = Path(config.model_path) / "smoke_metadata.json"
    if smoke_path.exists():
        receipt["smoke_model_metadata"] = json.loads(smoke_path.read_text(encoding="utf-8"))
        if not args.initial_checkpoint:
            receipt["baseline"] = "random_tiny_encoder_and_random_task_heads"
    config.update_labels(receipt["ner_labels"], receipt["re_labels"])
    config.epochs, config.batch_size, config.max_length = args.epochs, args.batch_size, args.max_length
    config.learning_rate, config.classifier_lr = args.learning_rate, args.classifier_lr
    config.early_stopping_patience = args.patience
    config.ner_threshold, config.re_threshold = args.ner_threshold, args.re_threshold
    config.re_max_distance = args.re_max_distance
    config.output_dir = str(args.output_dir.resolve())
    config.best_model_path = str((args.output_dir / "best_model").resolve())
    config.use_lora = True
    device = torch.device(config.device)
    tokenizer = config.tokenizer
    token_audit = {name: audit_tokens(rows, tokenizer, args.max_length)
                   for name, rows in (("train", train), ("validation", validation), ("test", test))}
    write_json(args.output_dir / "token_audit.json", token_audit)
    expected = label_mappings(receipt, config.base_model_name)
    groups = read_label_groups(args.label_config)
    receipt.update({
        "runtime_versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "base_model_artifacts": artifact_hashes(Path(config.model_path)),
        "effective_settings": {name: getattr(config, name) for name in (
            "device", "learning_rate", "classifier_lr", "weight_decay", "warmup_ratio", "max_grad_norm",
            "early_stopping_patience", "early_stopping_min_delta", "hidden_dropout_prob",
            "ner_loss_weight", "re_loss_weight", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")},
    })

    def new_model():
        return ModernBERTForNERRE(config.model_path, config.num_ner_labels, config.num_re_labels,
                                 hidden_dropout_prob=config.hidden_dropout_prob)

    def load_checkpoint(path, trainable=False):
        is_lora = verify_checkpoint(path, expected)
        loaded = new_model()
        if is_lora:
            loaded.bert = PeftModel.from_pretrained(loaded.bert, str(path), is_trainable=trainable)
            heads = torch.load(path / "classification_heads.pt", map_location="cpu", weights_only=True)
            if heads["num_ner_labels"] != config.num_ner_labels or heads["num_re_labels"] != config.num_re_labels:
                raise ValueError("Saved classifier dimensions disagree with frozen label mapping")
            for name in ("ner_classifier", "re_classifier", "distance_embedding"):
                getattr(loaded, name).load_state_dict(heads[name], strict=True)
        else:
            state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            # Training-only class weights are recomputed from this train split.
            state = {key: value for key, value in state.items() if key not in ("ner_label_weights", "re_label_weights")}
            loaded.load_state_dict(state, strict=True)
            if trainable:
                loaded.bert = get_peft_model(loaded.bert, lora_config())
        return loaded.to(device)

    def lora_config():
        return LoraConfig(task_type=TaskType.TOKEN_CLS, r=config.lora_r, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules, bias="none")

    def predict_rows(model, rows):
        predictions = []
        model.eval()
        with torch.no_grad():
            for row in rows:
                encoding = tokenizer(row["text"], truncation=False, return_offsets_mapping=True, return_tensors="pt")
                offsets = encoding["offset_mapping"][0].tolist()
                mask = encoding["attention_mask"].clone()
                # Existing decoder must not emit entities on CLS/SEP zero offsets.
                for index, (start, end) in enumerate(offsets):
                    if start == end:
                        mask[0, index] = 0
                result = model.predict(encoding["input_ids"].to(device), mask.to(device),
                                       config.ner_id2label, config.re_id2label,
                                       args.ner_threshold, args.re_threshold)[0]
                entities = []
                for entity in result["entities"]:
                    start, end = offsets[entity["start"]][0], offsets[entity["end"] - 1][1]
                    if not 0 <= start < end <= len(row["text"]):
                        raise ValueError("Decoder returned invalid character offsets")
                    entities.append({**entity, "start": start, "end": end})
                by_id = {e["id"]: e for e in entities}
                relations = []
                for relation in result["relations"]:
                    source, target = by_id[relation["from_id"]], by_id[relation["to_id"]]
                    gap = max(source["start"], target["start"]) - min(source["end"], target["end"])
                    if args.re_max_distance <= 0 or gap <= args.re_max_distance:
                        relations.append(relation)
                predictions.append({"task_id": row["task_id"], "text": row["text"], "entities": entities,
                                    "relations": relations, "status": "ok"})
        return predictions

    if args.initial_checkpoint:
        model = load_checkpoint(args.initial_checkpoint, trainable=True)
    else:
        model = new_model()
        model.bert = get_peft_model(model.bert, lora_config())
        model.to(device)
    receipt["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not any(p.requires_grad and "lora_" in name for name, p in model.named_parameters()):
        raise ValueError("No trainable LoRA parameters; refuse classifier-only training")
    receipt["status"] = "before_evaluation_started"
    write_json(args.output_dir / "experiment_receipt.json", receipt)
    before = predict_rows(model, test)
    write_json(args.output_dir / "before_predictions.json", before)
    before_report = joint_report(test, {"before": before}, groups, receipt["re_labels"])
    write_json(args.output_dir / "before_metrics.json", before_report)

    dataset = NERREDataset(train, tokenizer, args.max_length, config.ner_label2id, config.re_label2id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model.register_buffer("ner_label_weights", compute_label_weights(train))
    classifier, encoder = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (classifier if any(key in name for key in ("ner_classifier", "re_classifier", "distance_embedding")) else encoder).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": config.learning_rate}, {"params": classifier, "lr": config.classifier_lr},
    ], weight_decay=config.weight_decay)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    best_score, stale = -math.inf, 0
    history = []
    receipt["status"] = "training_started"
    write_json(args.output_dir / "experiment_receipt.json", receipt)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(model, loader, optimizer, scheduler, device, epoch, scaler=scaler)
        predicted = predict_rows(model, validation)
        val = joint_report(validation, {"after": predicted}, groups, receipt["re_labels"])
        ner_f1 = val["runs"]["after"]["overall"]["micro"]["f1"]
        re_f1 = val["relations"]["after"]["overall"]["micro"]["f1"]
        # Absent targets do not contribute a fabricated F1 to selection.
        supported = [value for value in (ner_f1, re_f1) if value is not None]
        if not supported:
            raise ValueError("Validation requires at least one supported NER or RE target")
        score = sum(supported) / len(supported)
        selected = score > best_score + config.early_stopping_min_delta
        history.append({"epoch": epoch, "train": train_metrics, "validation_ner_micro_f1": ner_f1,
                        "validation_end_to_end_re_micro_f1": re_f1, "selection_score": score,
                        "checkpoint_selected": selected, "duration_seconds": time.perf_counter() - epoch_started})
        with (args.output_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
        logging.info("Epoch %s: NER=%s RE=%s selection=%s saved=%s", epoch, ner_f1, re_f1, score, selected)
        if selected:
            best_score, stale = score, 0
            _save_model(model, config.best_model_path, is_lora=True, best_f1=score,
                        optimizer=optimizer, scheduler=scheduler, epoch=epoch, epochs_trained=epoch)
            write_json(Path(config.best_model_path) / "label_mappings.json", expected)
        else:
            stale += 1
            if stale >= args.patience:
                break

    receipt["saved_checkpoint_selection"] = [item for item in history if item["checkpoint_selected"]][-1]
    receipt["status"] = "checkpoint_saved_after_evaluation_pending"
    write_json(args.output_dir / "experiment_receipt.json", receipt)
    # Free optimizer and last-epoch model before reloading the selected checkpoint.
    del optimizer, scheduler, scaler, model, classifier, encoder, parameter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = load_checkpoint(Path(config.best_model_path))
    after = predict_rows(model, test)
    write_json(args.output_dir / "after_predictions.json", after)
    report = joint_report(test, {"before": before, "after": after}, groups, receipt["re_labels"])
    report["experiment"] = {"baseline": receipt["baseline"], "dataset_kind": receipt["dataset_kind"], "selection": receipt["selection"],
                            "test_sha256": receipt["inputs"]["test"]["sha256"],
                            "settings": receipt["requested_settings"]}
    write_json(args.output_dir / "comparison.json", report)
    (args.output_dir / "comparison.md").write_text(joint_markdown(report), encoding="utf-8")
    receipt["checkpoint_artifacts"] = artifact_hashes(Path(config.best_model_path))
    receipt["evaluation_artifacts"] = {name: sha256(args.output_dir / name) for name in (
        "before_predictions.json", "before_metrics.json", "after_predictions.json", "comparison.json", "comparison.md")}
    verify_inputs(args, receipt)
    receipt["experiment_wall_time_seconds"] = time.perf_counter() - started
    receipt["status"] = "completed_best_checkpoint_reloaded_and_test_evaluated"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("splits-dir", "reference", "label-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--base-model", default="chinese-modernbert-large-wwm")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--classifier-lr", type=float, default=5e-4)
    parser.add_argument("--ner-threshold", type=float, default=.5)
    parser.add_argument("--re-threshold", type=float, default=.5)
    parser.add_argument("--re-max-distance", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("Use a new output directory; experiments are never overwritten or resumed")
        if min(args.epochs, args.batch_size, args.max_length, args.patience) < 1:
            raise ValueError("Epochs, batch size, max length and patience must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (args.learning_rate, args.classifier_lr)):
            raise ValueError("Learning rates must be finite and positive")
        if any(not 0 <= v <= 1 for v in (args.ner_threshold, args.re_threshold)) or args.re_max_distance < 0:
            raise ValueError("Thresholds must be in [0,1]; relation distance must be nonnegative")
        train, validation, receipt = prepare_frozen_training(args.splits_dir, args.reference, args.label_config)
        test_raw = json.loads((args.splits_dir / "test.json").read_text(encoding="utf-8-sig"))
        # Validate test relation types without imposing training BIO overlap policy.
        for row in test_raw:
            relation_keys(row, receipt["re_labels"], reference=True)
        # Explicit arrays prevent accidentally treating omitted relations as negatives.
        source = json.loads(args.reference.read_text(encoding="utf-8-sig"))
        verify_inputs(args, receipt)
        for row in source:
            relation_keys(row, receipt["re_labels"], reference=True)
        if not any(row["entities"] or row["relations"] for row in validation):
            raise ValueError("Validation must contain supported targets for checkpoint selection")
        if not any(row["entities"] or row["relations"] for row in train):
            raise ValueError("Train split contains no positive targets")
        if not test_raw:
            raise ValueError("Nonempty held-out test split required")
        # Keep held-out spans unchanged; scorer resolves endpoint direction.
        test = [{**row, "task_id": str(row["task_id"])} for row in test_raw]
        expected = label_mappings(receipt, args.base_model)
        checkpoint_hashes = None
        if args.initial_checkpoint:
            verify_checkpoint(args.initial_checkpoint, expected)
            checkpoint_hashes = artifact_hashes(args.initial_checkpoint)
        smoke_path = ROOT / "pretrained_models" / args.base_model / "smoke_metadata.json"
        baseline = "existing_task_checkpoint" if args.initial_checkpoint else (
            "random_tiny_encoder_and_random_task_heads" if smoke_path.is_file() else "pretrained_encoder_random_task_heads")
        receipt.update({
            "protocol_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "data_preflight_passed_training_not_run",
            "baseline": baseline,
            "dataset_kind": "synthetic_smoke" if all(row.get("dataset_kind") == "synthetic_smoke" for row in source) else "unverified",
            "initial_checkpoint_artifacts": checkpoint_hashes,
            "selection": "mean_of_supported_validation_strict_ner_and_end_to_end_re_micro_f1",
            "requested_settings": {name: getattr(args, name) for name in (
                "base_model", "epochs", "batch_size", "max_length", "seed", "patience", "learning_rate",
                "classifier_lr", "ner_threshold", "re_threshold", "re_max_distance")},
            "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                Path(__file__), ROOT / "modernbert_ml_backend/train_ner_re.py", ROOT / "modernbert_ml_backend/model.py",
                ROOT / "modernbert_ml_backend/modeling/network.py", ROOT / "modernbert_ml_backend/modeling/inference.py",
                ROOT / "modernbert_ml_backend/training/trainer.py", ROOT / "modernbert_ml_backend/label_studio_backend/service.py",
                ROOT / "modernbert_ml_backend/backend/service.py",
                ROOT / "modernbert_ml_backend/config.py", ROOT / "emr_annotation/training_data/frozen.py",
                ROOT / "emr_annotation/training_data/token_audit.py", ROOT / "emr_annotation/evaluation/ner_re.py",
                ROOT / "emr_annotation/evaluation/entity_predictions.py", ROOT / "emr_annotation/annotation/schema.py")},
        })
        receipt["notes"] = [
            "Hashes/splits do not certify completed medical adjudication, exhaustive negatives, near-duplicate separation or unseen test history.",
            "check-only does not run tokenizer, models, training or evaluation.",
            "Runtime rejects all overlength texts, nonrepresentable boundaries and token collisions before any evaluation.",
            "Inference retains existing model decoding, same-type-pair exclusion and character-distance filter; zero-offset tokens are masked.",
            "Attributes/case choices remain in source and are not learned/evaluated by current NER+RE heads.",
            "No checkpoint is selected from test scores; before/after test settings are identical and fixed in advance.",
            "A random-task-head baseline is an initialization diagnostic, not a previously trained task model.",
            "A fixed seed does not guarantee bitwise reproducibility across hardware.",
        ]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "experiment_receipt.json", receipt)
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"Preflight: train={len(train)}, validation={len(validation)}, test={len(test)}; baseline={receipt['baseline']}")
    if args.check_only:
        return 0
    logger = logging.getLogger()
    previous_level = logger.level
    handlers = [logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")]
    logger.setLevel(logging.INFO)
    for handler in handlers:
        logger.addHandler(handler)
    try:
        run_experiment(args, train, validation, test, receipt)
    except Exception as exc:
        receipt.update({"status": "experiment_failed", "failure_type": type(exc).__name__})
        write_json(args.output_dir / "experiment_receipt.json", receipt)
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
    write_json(args.output_dir / "experiment_receipt.json", receipt)
    print(f"Best checkpoint reloaded; before/after report: {args.output_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
