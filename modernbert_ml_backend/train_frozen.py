"""Train verified patient splits without resplitting or oversampling.

Run as a script to retain the backend's existing model/config import behavior.
--check-only validates files without importing ML dependencies or loading weights.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
from scripts.frozen_training_data import prepare_frozen_training


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--base-model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if min(args.epochs, args.batch_size, args.max_length) < 1:
            raise ValueError("Epochs, batch size and maximum length must be positive")
        if args.output_dir.exists():
            raise ValueError("Choose a new output directory; existing experiments are never overwritten or resumed")
        train, validation, receipt = prepare_frozen_training(args.splits_dir, args.reference, args.label_config)
        receipt.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "runner_sha256": file_sha256(Path(__file__)),
                        "converter_sha256": file_sha256(ROOT / "scripts/frozen_training_data.py"),
                        "trainer_sha256": file_sha256(ROOT / "modernbert_ml_backend/train_ner_re.py"),
                        "config_sha256": file_sha256(ROOT / "modernbert_ml_backend/config.py"),
                        "requested_settings": {"base_model": args.base_model, "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length, "seed": args.seed}})
        args.output_dir.mkdir(parents=True, exist_ok=False)
        receipt_path = args.output_dir / "training_receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"Frozen data: train={len(train)}, validation={len(validation)}, test={receipt['splits']['test']['tasks']}; schema={receipt['schema_id']}")
    if args.check_only:
        print("Data preflight passed; no ML imports, tokenizer, model or training executed")
        return 0
    try:
        # Local script imports intentionally preserve the existing backend runtime.
        from config import config
        from train_ner_re import train_from_splits
        import peft  # Require LoRA; do not silently fall back to full fine-tuning.
        import torch
        import transformers

        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")])
        if args.base_model:
            config.update_model_paths(args.base_model, None)
        if not Path(config.model_path).is_dir():
            raise ValueError("Configured local base-model directory does not exist")
        config.update_labels(receipt["ner_labels"], receipt["re_labels"])
        config.epochs, config.batch_size, config.max_length = args.epochs, args.batch_size, args.max_length
        config.use_lora = True
        config.output_dir = str(args.output_dir.resolve())
        config.best_model_path = str((args.output_dir / "best_model").resolve())
        receipt["status"] = "training_started"
        receipt["effective_settings"] = {"base_model": config.base_model_name, "base_model_path": config.model_path,
                                          "device": config.device,
                                          **{name: getattr(config, name) for name in (
                                              "learning_rate", "classifier_lr", "weight_decay", "warmup_ratio", "max_grad_norm",
                                              "early_stopping_patience", "early_stopping_min_delta", "hidden_dropout_prob",
                                              "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")},
                                          **{k: receipt["requested_settings"][k] for k in ("epochs", "batch_size", "max_length", "seed")}}
        receipt["runtime_versions"] = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history_path = args.output_dir / "validation_history.jsonl"
        def record_epoch(metrics):
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        _, selection_score = train_from_splits(train, validation, seed=args.seed, metrics_callback=record_epoch)
        best_path = Path(config.best_model_path)
        required = ("adapter_config.json", "classification_heads.pt", "checkpoint.pt")
        if not all((best_path / name).is_file() for name in required) or not any((best_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError("Training returned without a complete saved LoRA checkpoint")
        (best_path / "label_mappings.json").write_text(json.dumps({"schema_id": receipt["schema_id"], "ner_label2id": config.ner_label2id, "re_label2id": config.re_label2id}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
        selected = [epoch for epoch in history if epoch["checkpoint_selected"]]
        if not selected:
            raise ValueError("No saved checkpoint selection recorded")
        receipt.update({"status": "training_completed_checkpoint_saved_not_independently_evaluated",
                        "trainer_reported_best_selection_score": selection_score,
                        "saved_checkpoint_selection": selected[-1],
                        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in sorted(best_path.iterdir()) if p.is_file()}})
    except Exception as exc:
        receipt["status"] = "training_failed"
        receipt["failure_type"] = type(exc).__name__
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Checkpoint saved; final entity scores and service measurements remain separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
