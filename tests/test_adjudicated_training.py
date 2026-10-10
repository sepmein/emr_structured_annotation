import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from modernbert_ml_backend.train_adjudicated import main, label_mappings, verify_checkpoint
from emr_annotation.training_data.token_audit import audit_tokens
from scripts.build_training_smoke_fixture import synthetic_rows
from emr_annotation.evaluation.ner_re import joint_report, relation_keys, main as score_main
from emr_annotation.evaluation.entity_predictions import read_label_groups
from emr_annotation.training_data.frozen import prepare_frozen_training
from emr_annotation.training_data.split_reference import split_reference

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "label_studio/pneumonia_config.xml"


def row(n=1):
    return {"task_id": str(n), "patient_id": f"p{n}", "text": f"体温38℃虚构{n}",
            "entities": [{"id": "a", "start": 0, "end": 2, "label": "体温"},
                         {"id": "b", "start": 2, "end": 5, "label": "数值"}],
            "relations": [{"from_id": "a", "to_id": "b", "type": "测量"}]}


def fixture(root):
    rows = [row(n) for n in range(9)]
    reference = root / "reference.json"
    reference.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    splits, manifest = split_reference(rows, read_label_groups(CONFIG), (.6, .2, .2), 42)
    manifest.update(source_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    label_config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    directory = root / "splits"
    directory.mkdir()
    for name, payload in {**splits, "split_manifest": manifest}.items():
        (directory / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return ["--splits-dir", str(directory), "--reference", str(reference), "--label-config", str(CONFIG),
            "--output-dir", str(root / "run")], directory, reference


class JointScoringTests(unittest.TestCase):
    def score(self, gold, prediction):
        return joint_report([gold], {"before": [prediction]}, {"labels": ["体温", "数值"]}, ["测量"])

    def test_ids_irrelevant_but_endpoints_type_and_direction_are_strict(self):
        gold = row()
        predicted = copy.deepcopy(gold)
        predicted["entities"][0]["id"], predicted["entities"][1]["id"] = "pred0", "pred1"
        predicted["relations"][0].update(from_id="pred0", to_id="pred1")
        self.assertEqual(self.score(gold, predicted)["relations"]["before"]["overall"]["micro"]["f1"], 1)
        for mode in ("direction", "boundary", "label"):
            broken = copy.deepcopy(predicted)
            if mode == "direction": broken["relations"][0]["direction"] = "left"
            elif mode == "boundary": broken["entities"][0]["end"] = 1
            else: broken["entities"][0]["label"] = "数值"
            m = self.score(gold, broken)["relations"]["before"]["overall"]["micro"]
            self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    def test_missing_entities_duplicate_predictions_and_negatives(self):
        gold = row()
        missing = {**gold, "entities": [], "relations": []}
        self.assertEqual(self.score(gold, missing)["relations"]["before"]["overall"]["micro"]["fn"], 1)
        duplicate = copy.deepcopy(gold)
        duplicate["relations"] *= 2
        self.assertEqual(self.score(gold, duplicate)["relations"]["before"]["overall"]["micro"]["fp"], 1)
        # False positives from a negative case remain in the overall denominator.
        negative = {**row(2), "entities": [], "relations": []}
        report = joint_report([gold, negative], {"after": [gold, row(2)]}, {"labels": ["体温", "数值"]}, ["测量", "起始时间"])
        m = report["relations"]["after"]["overall"]["micro"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))
        self.assertIsNone(report["relations"]["after"]["per_label"]["起始时间"]["f1"])

    def test_missing_failed_tasks_and_invalid_contracts(self):
        gold = row()
        for predictions in ([], [{**gold, "status": "failed", "entities": [], "relations": []}]):
            r = joint_report([gold], {"before": predictions}, {"labels": ["体温", "数值"]}, ["测量"])
            self.assertEqual(r["relations"]["before"]["overall"]["micro"]["fn"], 1)
        for mode in ("missing_type", "dangling", "duplicate_gold", "missing_array", "text_changed"):
            broken = copy.deepcopy(gold)
            if mode == "missing_type": del broken["relations"][0]["type"]
            elif mode == "dangling": broken["relations"][0]["to_id"] = "unknown"
            elif mode == "duplicate_gold": broken["relations"] *= 2
            elif mode == "missing_array": del broken["relations"]
            else: broken["text"] += "不同"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                if mode == "duplicate_gold": relation_keys(broken, ["测量"], reference=True)
                else: self.score(gold, broken)

    def test_bidirectional_gold_and_overall_change(self):
        gold = row()
        gold["relations"][0]["direction"] = "bi"
        r = self.score(gold, row())
        self.assertEqual(r["relations"]["before"]["overall"]["micro"]["fn"], 1)
        empty = {**row(), "entities": [], "relations": []}
        r = joint_report([row()], {"before": [empty], "after": [row()]}, {"labels": ["体温", "数值"]}, ["测量"])
        self.assertEqual(r["overall_f1_change"], {"entities": 1, "relations": 1})

    def test_cli_hashes_inputs_and_writes_all_configured_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, prediction = root / "reference.json", root / "before.json"
            for path in (reference, prediction): path.write_text(json.dumps([row()]), encoding="utf-8")
            args = ["--reference", str(reference), "--before", str(prediction), "--label-config", str(CONFIG), "--output-dir", str(root / "score")]
            self.assertEqual(score_main(args), 0)
            report = json.loads((root / "score/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]["before"]["per_label"]), 49)
            self.assertEqual(len(report["relations"]["before"]["per_label"]), 5)
            self.assertEqual(report["inputs"]["reference"]["sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): score_main(args)


class TokenAuditTests(unittest.TestCase):
    def test_smoke_fixture_uses_real_schema_and_all_relation_types(self):
        rows = synthetic_rows(144)
        groups = read_label_groups(ROOT / "label_studio/pneumonia_config.global-single.xml")
        splits, _ = split_reference(rows, groups, (.7, .15, .15), 42)
        for records in splits.values():
            report = joint_report(records, {"before": records}, groups,
                                  ["发生时间", "起始时间", "结束时间", "持续时长", "测量"])
            self.assertEqual(report["runs"]["before"]["overall"]["supported_label_count"], 5)
            self.assertEqual(report["relations"]["before"]["overall"]["supported_label_count"], 5)
        self.assertTrue(all(row["dataset_kind"] == "synthetic_smoke" for row in rows))

    def test_rejects_truncation_boundary_and_token_collisions(self):
        sample = row()
        valid = lambda text, **kwargs: {"input_ids": list(range(len(text) + 2)), "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]}
        self.assertEqual(audit_tokens([sample], valid, 100)["entities"], 2)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            audit_tokens([sample], valid, 3)
        boundary = lambda text, **kwargs: {"input_ids": [1, 2, 3], "offset_mapping": [(0, 3), (3, 5), (5, len(text))]}
        with self.assertRaisesRegex(ValueError, "boundary"):
            audit_tokens([sample], boundary, 100)
        overlapping = copy.deepcopy(sample)
        overlapping["entities"][0]["end"] = 3
        with self.assertRaisesRegex(ValueError, "share tokens"):
            audit_tokens([overlapping], valid, 100)
        negative = {**sample, "entities": [], "relations": []}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            audit_tokens([negative], valid, 3)


class ExperimentTests(unittest.TestCase):
    def test_check_only_with_no_site_packages_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _, _ = fixture(root)
            run = subprocess.run([sys.executable, "-S", str(ROOT / "modernbert_ml_backend/train_adjudicated.py"), *args, "--check-only"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads((root / "run/experiment_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["baseline"], "pretrained_encoder_random_task_heads")
            for path in ("modeling/network.py", "modeling/inference.py", "label_studio_backend/service.py", "backend/service.py", "training/trainer.py", "model.py", "train_ner_re.py"):
                actual = ROOT / "modernbert_ml_backend" / path
                self.assertEqual(receipt["code_sha256"][str(actual.relative_to(ROOT))], hashlib.sha256(actual.read_bytes()).hexdigest())
            self.assertFalse((root / "run/before_metrics.json").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*args, "--check-only"])

    def test_checkpoint_mapping_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            args, directory, source = fixture(path)
            _, _, receipt = prepare_frozen_training(directory, source, CONFIG)
            expected = label_mappings(receipt, "base")
            target = path / "checkpoint"
            target.mkdir()
            (target / "label_mappings.json").write_text(json.dumps(expected), encoding="utf-8")
            (target / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_checkpoint(target, expected)
            for name in ("classification_heads.pt", "adapter_model.safetensors"):
                (target / name).write_bytes(b"fake")
            self.assertTrue(verify_checkpoint(target, expected))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_checkpoint(target, {**expected, "base_model_name": "other"})

    def test_failure_recorded_without_completion_or_fake_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _, _ = fixture(root)
            with patch("modernbert_ml_backend.train_adjudicated.run_experiment", side_effect=RuntimeError("mock failure")), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): main(args)
            receipt = json.loads((root / "run/experiment_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "experiment_failed")
            self.assertFalse((root / "run/comparison.json").exists())

    def test_actual_experiment_loop_selects_validation_and_reloads_best(self):
        # Run the real orchestration with tiny simulated ML modules; no GPU/model.
        class Tensor:
            def __init__(self, value): self.value = value
            def __getitem__(self, key): return Tensor(self.value[key])
            def __setitem__(self, key, value): self.value[key[0]][key[1]] = value
            def clone(self): return Tensor(copy.deepcopy(self.value))
            def tolist(self): return self.value
            def to(self, device): return self
        def tokenizer(text, **kwargs):
            offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
            result = {"input_ids": list(range(len(offsets))), "attention_mask": [1] * len(offsets), "offset_mapping": offsets}
            return {k: Tensor([v]) for k, v in result.items()} if kwargs.get("return_tensors") else result
        events, saved = [], {}
        parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)
        class Head:
            def __init__(self, owner): self.owner = owner
            def load_state_dict(self, state, **kwargs): self.owner.stage = state["stage"]
        class Model:
            def __init__(self, *args, **kwargs):
                self.stage, self.bert = 0, object()
                for name in ("ner_classifier", "re_classifier", "distance_embedding"): setattr(self, name, Head(self))
            def to(self, device): return self
            def eval(self): return self
            def register_buffer(self, *args): pass
            def named_parameters(self): return [("bert.lora_A", parameter), ("ner_classifier", parameter)]
            def parameters(self): return [parameter]
            def predict(self, *args):
                events.append(("predict", self.stage))
                if self.stage != 1: return [{"entities": [], "relations": []}]
                return [{"entities": [{"id": "x", "start": 1, "end": 3, "label": "体温"}, {"id": "y", "start": 3, "end": 6, "label": "数值"}],
                         "relations": [{"from_id": "x", "to_id": "y", "type": "测量"}]}]
        cfg = SimpleNamespace(
            epochs=3, batch_size=4, max_length=1024, learning_rate=5e-5, classifier_lr=5e-4, device="cpu",
            weight_decay=.01, warmup_ratio=.1, max_grad_norm=1, early_stopping_min_delta=.001,
            early_stopping_patience=5, hidden_dropout_prob=.1, ner_loss_weight=1, re_loss_weight=2,
            lora_r=16, lora_alpha=32, lora_dropout=.1, lora_target_modules=["Wqkv"], tokenizer=tokenizer)
        def update_labels(ner, re):
            mappings = label_mappings({"ner_labels": ner, "re_labels": re, "schema_id": "unused"}, "base")
            for key in ("ner_label2id", "re_label2id"): setattr(cfg, key, mappings[key])
            cfg.num_ner_labels, cfg.num_re_labels = len(cfg.ner_label2id), len(cfg.re_label2id)
            cfg.ner_id2label, cfg.re_id2label = {}, {}
        cfg.update_labels = update_labels
        def epoch(model, *args, **kwargs):
            events.append(("train", model.stage)); model.stage += 1
            return {"loss": 0.5}
        def save(model, path, **kwargs):
            events.append(("save", model.stage))
            target = Path(path); target.mkdir(exist_ok=True)
            for name in ("classification_heads.pt", "adapter_config.json", "adapter_model.safetensors", "checkpoint.pt"):
                (target / name).write_text("{}")
            saved["heads"] = {"num_ner_labels": cfg.num_ner_labels, "num_re_labels": cfg.num_re_labels,
                              **{name: {"stage": model.stage} for name in ("ner_classifier", "re_classifier", "distance_embedding")}}
        modules = {}
        def module(name, **attrs):
            obj = ModuleType(name)
            obj.__dict__.update(attrs); modules[name] = obj
            return obj
        torch = module("torch", __version__="fake", manual_seed=lambda n: None, device=lambda d: SimpleNamespace(type=d),
                       cuda=SimpleNamespace(is_available=lambda: False), no_grad=contextlib.nullcontext,
                       optim=SimpleNamespace(AdamW=lambda *a, **k: None), load=lambda *a, **k: saved["heads"])
        module("torch.utils")
        module("torch.utils.data", DataLoader=lambda data, **kwargs: data)
        module("transformers", __version__="fake", get_linear_schedule_with_warmup=lambda *a: None)
        def load_adapter(*args, **kwargs):
            events.append(("load_trainable", kwargs["is_trainable"]))
            return object()
        module("peft", __version__="fake", LoraConfig=lambda **kwargs: kwargs, get_peft_model=lambda base, cfg: base,
               PeftModel=SimpleNamespace(from_pretrained=load_adapter), TaskType=SimpleNamespace(TOKEN_CLS="TOKEN_CLS"))
        module("config", config=cfg)
        module("model", ModernBERTForNERRE=Model)
        module("modeling", __path__=[])
        module("modeling.network", ModernBERTForNERRE=Model)
        module("train_ner_re", NERREDataset=lambda data, *a: data, collate_fn=None,
               compute_label_weights=lambda rows: None, train_epoch=epoch, _save_model=save)
        for warm_start in (False, True):
            with self.subTest(warm_start=warm_start), tempfile.TemporaryDirectory() as tmp:
                self.run_simulated_loop(Path(tmp), fixture, cfg, modules, events, saved, warm_start)

    def run_simulated_loop(self, root, make_fixture, cfg, modules, events, saved, warm_start):
        events.clear()
        args, directory, source = make_fixture(root)
        base = root / "base"; base.mkdir(); (base / "config.json").write_text("{}")
        def update_paths(name, scope): cfg.model_path, cfg.base_model_name = str(base), name
        cfg.update_model_paths = update_paths
        if warm_start:
            target = root / "initial"
            target.mkdir()
            _, _, receipt = prepare_frozen_training(directory, source, CONFIG)
            mapping = label_mappings(receipt, "chinese-modernbert-large-wwm")
            (target / "label_mappings.json").write_text(json.dumps(mapping), encoding="utf-8")
            for name in ("adapter_config.json", "adapter_model.safetensors", "classification_heads.pt"):
                (target / name).write_text("{}")
            saved["heads"] = {"num_ner_labels": len(mapping["ner_label2id"]), "num_re_labels": len(mapping["re_label2id"]),
                              **{name: {"stage": 0} for name in ("ner_classifier", "re_classifier", "distance_embedding")}}
            args += ["--initial-checkpoint", str(target)]
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([*args, "--epochs", "3", "--patience", "1"]), 0)
        report = json.loads((root / "run/comparison.json").read_text(encoding="utf-8"))
        receipt = json.loads((root / "run/experiment_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(report["overall_f1_change"], {"entities": 1, "relations": 1})
        self.assertEqual(receipt["saved_checkpoint_selection"]["epoch"], 1)
        self.assertEqual(receipt["status"], "completed_best_checkpoint_reloaded_and_test_evaluated")
        self.assertEqual([stage for event, stage in events if event == "train"], [0, 1])
        self.assertEqual([flag for event, flag in events if event == "load_trainable"], [True, False] if warm_start else [False])
        self.assertEqual(receipt["baseline"], "existing_task_checkpoint" if warm_start else "pretrained_encoder_random_task_heads")
        # Last epoch=2 is bad, but final predictions use reloaded epoch=1.
        self.assertEqual(events[-1], ("predict", 1))


if __name__ == "__main__":
    unittest.main()
