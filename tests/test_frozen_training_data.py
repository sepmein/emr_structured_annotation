import ast
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest

from scripts.frozen_training_data import convert_rows, prepare_frozen_training
from scripts.split_evaluation_reference import split_reference
from scripts.evaluate_entity_predictions import read_label_groups
from modernbert_ml_backend.train_frozen import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "label_studio/pneumonia_config.xml"
TRAINER = ROOT / "modernbert_ml_backend/train_ner_re.py"


def row(task_id, text, positive=True):
    return {"task_id": str(task_id), "patient_id": f"p{task_id}", "text": text,
            "entities": [{"id": "a", "start": 0, "end": 2, "label": "发热"}] if positive else [],
            "relations": [], "attributes": [], "case_choices": {"case_decision": "待专业复核"}}


def fixture(root, rows=None):
    rows = rows or [row(1, "发热一天"), row(2, "发热两天"), row(3, "一般情况可", False), row(4, "普通复诊", False), row(5, "发热三天"), row(6, "常规随访", False)]
    reference = root / "reference.json"
    reference.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    splits, manifest = split_reference(rows, read_label_groups(CONFIG), (.5, .25, .25), 42)
    manifest.update({"source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "label_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()})
    directory = root / "splits"
    directory.mkdir()
    for name, payload in {**splits, "split_manifest": manifest}.items():
        (directory / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return directory, reference, splits


def load_trainer_function(name, namespace):
    # Exercise the actual function body without importing heavyweight ML modules.
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return namespace[name]


class FrozenTrainingDataTests(unittest.TestCase):
    def test_reviewed_splits_keep_negatives_metadata_source_and_xml_order(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, source, splits = fixture(Path(temp))
            train, validation, receipt = prepare_frozen_training(directory, source, CONFIG)
            self.assertEqual([r["task_id"] for r in train], [r["task_id"] for r in splits["train"]])
            self.assertEqual([r["task_id"] for r in validation], [r["task_id"] for r in splits["validation"]])
            self.assertEqual(sum(not r["entities"] for r in train), sum(not r["entities"] for r in splits["train"]))
            self.assertFalse({r["task_id"] for r in train+validation} & {r["task_id"] for r in splits["test"]})
            self.assertEqual(len(receipt["ner_labels"]), 49)
            self.assertEqual(len(receipt["re_labels"]), 5)
            self.assertEqual(receipt["ner_labels"][:2], ["发热", "鼻翼煽动"])
            self.assertNotIn("发热一天", json.dumps(receipt, ensure_ascii=False))
            self.assertIn("case_choices", json.loads(source.read_text(encoding="utf-8"))[0])

    def test_changed_entities_manifest_and_patient_leakage_rejected(self):
        for mode in ("entity_changed", "assignment_changed", "patient_overlap", "reference_hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory, source, splits = fixture(Path(temp))
                if mode == "entity_changed":
                    splits["train"][0]["entities"] = [] if splits["train"][0]["entities"] else [{"start": 0,"end": 2,"label": "发热"}]
                    (directory/"train.json").write_text(json.dumps(splits["train"]), encoding="utf-8")
                elif mode == "reference_hash":
                    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                elif mode == "assignment_changed":
                    p = directory / "split_manifest.json"
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    payload["assignments"][0]["split"] = "wrong"
                    p.write_text(json.dumps(payload), encoding="utf-8")
                else:
                    # Even a self-consistent source and manifest cannot mask patient leakage.
                    rows = json.loads(source.read_text(encoding="utf-8"))
                    ids = {splits["train"][0]["task_id"], splits["test"][0]["task_id"]}
                    for r in rows:
                        if r["task_id"] in ids:
                            r["patient_id"] = "same-patient"
                    source.write_text(json.dumps(rows), encoding="utf-8")
                    by_id = {r["task_id"]:r for r in rows}
                    for name, split in splits.items():
                        (directory/f"{name}.json").write_text(json.dumps([by_id[r["task_id"]] for r in split]), encoding="utf-8")
                    p = directory / "split_manifest.json"
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                    p.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prepare_frozen_training(directory, source, CONFIG)

    def test_direction_conversion_and_unsupported_pairs(self):
        sample = {"task_id": "1", "text": "发热休克", "entities": [{"id":"a","start":0,"end":2,"label":"发热"},{"id":"b","start":2,"end":4,"label":"休克"}],
                  "relations": [{"from_id":"a","to_id":"b","type":"test-relation","direction":"left"}]}
        out = convert_rows([sample], {"发热", "休克"}, ["test-relation"])[0]
        self.assertEqual(out["relations"], [{"from_id":"b","to_id":"a","type":"test-relation"}])
        sample["relations"][0]["direction"] = "bi"
        self.assertEqual(len(convert_rows([sample], {"发热", "休克"}, ["test-relation"])[0]["relations"]), 2)
        for mode in ("dangling", "duplicate", "duplicate_id", "overlap"):
            broken = copy.deepcopy(sample)
            if mode == "dangling": broken["relations"][0]["to_id"] = "missing"
            elif mode == "duplicate": broken["relations"].append(copy.deepcopy(broken["relations"][0]))
            elif mode == "duplicate_id": broken["entities"][1]["id"] = "a"
            else: broken["entities"][1]["start"] = 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                convert_rows([broken], {"发热", "休克"}, ["test-relation"])

    def test_check_only_runs_without_ml_imports_and_preserves_previous_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, source, _ = fixture(root)
            out = root / "check"
            args = ["--splits-dir",str(directory),"--reference",str(source),"--label-config",str(CONFIG),"--output-dir",str(out),"--check-only"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            receipt = json.loads((out/"training_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "data_preflight_passed_training_not_run")
            self.assertFalse((out/"best_model").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_train_from_splits_passes_only_explicit_data_without_oversampling(self):
        train, validation = [row(1,"发热"),row(2,"无目标",False)], [row(3,"复诊",False)]
        before = copy.deepcopy((train,validation))
        captured = {}
        def dataset(data,*args):
            return data
        def train_model(*args,**kwargs):
            captured["args"], captured["kwargs"] = args,kwargs
            return "model",0
        with tempfile.TemporaryDirectory() as temp:
            cfg = SimpleNamespace(ner_labels=["发热"],best_model_path=str(Path(temp)/"new"),tokenizer="unused",max_length=10,
                                  ner_label2id={},re_label2id={},batch_size=2,device="cpu")
            torch = SimpleNamespace(manual_seed=lambda _:None, cuda=SimpleNamespace(is_available=lambda:False),device=lambda d:d)
            ns = {"config":cfg,"os":os,"random":random,"torch":torch,"NERREDataset":dataset,"DataLoader":lambda data,**kwargs:(data,kwargs),
                  "collate_fn":None,"compute_label_weights":lambda rows:tuple(r["task_id"] for r in rows),"train_model":train_model}
            fn = load_trainer_function("train_from_splits",ns)
            self.assertEqual(fn(train,validation),("model",0))
            self.assertEqual(captured["args"][0][0],train)
            self.assertEqual(captured["args"][1][0],validation)
            self.assertEqual(captured["kwargs"]["ner_label_weights"],("1","2"))
            self.assertTrue(captured["kwargs"]["save_first_validation"])
            self.assertEqual((train,validation),before)
            with self.assertRaises(ValueError): fn(train,[])

    def test_first_zero_validation_checkpoint_saved_only_when_requested(self):
        class Model:
            def __init__(self,**kwargs): pass
            def to(self,device): pass
            def parameters(self): return []
        for flag, expected in ((False,0),(True,1)):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temp:
                saves,history = [],[]
                cfg = SimpleNamespace(model_path="base",num_ner_labels=3,num_re_labels=1,hidden_dropout_prob=.1,use_lora=False,
                    best_model_path=str(Path(temp)/"best"),learning_rate=.01,classifier_lr=.01,weight_decay=0,epochs=1,warmup_ratio=0,
                    output_dir=temp,early_stopping_min_delta=.001,early_stopping_patience=5)
                torch = SimpleNamespace(optim=SimpleNamespace(AdamW=lambda *a,**k:None))
                ns = {"config":cfg,"torch":torch,"os":os,"logger":logging.getLogger("test-frozen"),"ModernBERTForNERRE":Model,
                      "_load_model_if_exists":lambda *a:None,"get_linear_schedule_with_warmup":lambda *a,**k:None,
                      "train_epoch":lambda *a,**k:{"loss":0},"evaluate":lambda *a,**k:{"loss":0,"ner_f1":0,"re_f1":0},
                      "_save_model":lambda *a,**k:saves.append(k)}
                fn = load_trainer_function("train_model",ns)
                fn([1],SimpleNamespace(dataset=[1]),SimpleNamespace(type="cpu"),metrics_callback=history.append,save_first_validation=flag)
                self.assertEqual(len(saves),expected)
                self.assertEqual(history[0]["checkpoint_selected"],flag)


if __name__ == "__main__":
    unittest.main()
