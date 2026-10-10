"""Build fictional records and a random tiny ModernBERT for real pipeline testing.

No clinical records, downloaded weights or trained production models are used.
Use the independent modernbert_ml_backend/requirements.txt environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from emr_annotation.training_data.split_reference import split_reference
from emr_annotation.evaluation.entity_predictions import read_label_groups


def synthetic_rows(count):
    templates = [
        ("体温38.5℃", [("体温", "体温"), ("38.5℃", "数值")], "测量"),
        ("昨日出现发热", [("发热", "发热"), ("昨日", "时间表达")], "发生时间"),
        ("今晨开始喘息", [("喘息", "喘鸣喘息"), ("今晨", "时间表达")], "起始时间"),
        ("发热昨日结束", [("发热", "发热"), ("昨日", "时间表达")], "结束时间"),
        ("喘息持续三天", [("喘息", "喘鸣喘息"), ("三天", "时间表达")], "持续时长"),
        ("发热伴喘息", [("发热", "发热"), ("喘息", "喘鸣喘息")], None),
        ("体温37.1℃", [("体温", "体温"), ("37.1℃", "数值")], "测量"),
        ("一般情况良好", [], None),
    ]
    rows = []
    for index in range(count):
        phrase, spans, relation = templates[index % len(templates)]
        text = f"虚构记录{index:03d}：{phrase}。"
        entities = [{"id": f"e{i}", "start": text.index(surface), "end": text.index(surface) + len(surface),
                     "label": label, "text": surface} for i, (surface, label) in enumerate(spans)]
        rows.append({"task_id": f"synthetic-{index:03d}", "patient_id": f"synthetic-patient-{index:03d}",
                     "text": text, "entities": entities,
                     "relations": [{"from_id": "e0", "to_id": "e1", "type": relation}] if relation else [],
                     "attributes": [], "case_choices": {}, "dataset_kind": "synthetic_smoke"})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--tasks", type=int, default=144)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    base = ROOT / "pretrained_models" / args.base_model_name
    if args.output_dir.exists() or base.exists() or base.parent != ROOT / "pretrained_models" or args.tasks < 24:
        parser.error("Use fresh output/model directories, a single model-name component and >=24 tasks")

    import torch
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    torch.manual_seed(args.seed)
    rows = synthetic_rows(args.tasks)
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted({char for row in rows for char in row["text"]}))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                       cls_token="[CLS]", sep_token="[SEP]", model_max_length=128)
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=96,
                          num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
                          local_attention=16, pad_token_id=0, bos_token_id=2, eos_token_id=3,
                          cls_token_id=2, sep_token_id=3, reference_compile=False)
    model = ModernBertModel(cfg)
    metadata = {"dataset_kind": "synthetic_smoke", "encoder_initialization": "random",
                "seed": args.seed, "tasks": args.tasks, "encoder_parameters": sum(p.numel() for p in model.parameters()),
                "hidden_size": 64, "layers": 2, "attention_heads": 4,
                "notes": ["Eight repeated fictional templates; not a medical-generalization benchmark.",
                          "Random tiny encoder and character tokenizer; not the production Chinese pretrained encoder."]}
    base.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    (base / "smoke_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    xml = ROOT / "label_studio/pneumonia_config.global-single.xml"
    snapshot = args.output_dir / "label_config.xml"
    snapshot.write_bytes(xml.read_bytes())
    reference = args.output_dir / "reference.json"
    reference.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    splits, manifest = split_reference(rows, read_label_groups(snapshot), (.7, .15, .15), args.seed)
    manifest.update(source_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), label_config_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    directory = args.output_dir / "splits"
    directory.mkdir()
    for name, data in {**splits, "split_manifest": manifest}.items():
        (directory / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fixture_receipt.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_name": args.base_model_name, "encoder_parameters": metadata["encoder_parameters"],
                      "splits": {name: len(data) for name, data in splits.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
