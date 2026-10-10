"""Verify legacy callers, offline imports and synthetic CLI contracts after relocation."""
import hashlib
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "merge_emr_data": "data_preparation.merge_emr_data",
    "evaluate_entity_predictions": "evaluation.entity_predictions",
    "prepare_label_studio_evaluation": "annotation_analysis.label_studio_evaluation",
    "audit_double_annotations": "annotation_analysis.double_annotations",
    "frozen_training_data": "training_data.frozen",
    "split_evaluation_reference": "training_data.split_reference",
    "prepare_training_data": "training_data.preparation",
    "audit_training_tokens": "training_data.token_audit",
    "evaluate_ner_re": "evaluation.ner_re",
    "regex_entity_baseline": "evaluation.regex_entity_baseline",
    "benchmark_model_service": "evaluation.benchmark_model_service",
    "benchmark_chat_model": "evaluation.benchmark_chat_model",
    "compare_model_benchmarks": "evaluation.compare_model_benchmarks",
    "package_double_annotation_review": "adjudication.package_review",
}


def run_python(*args):
    return subprocess.run(
        [sys.executable, "-S", *map(str, args)], cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


class SharedCodeMigrationTests(unittest.TestCase):
    def test_all_legacy_modules_are_canonical_objects_in_both_import_orders(self):
        for old, new in MAPPING.items():
            with self.subTest(module=old):
                canonical = importlib.import_module("emr_annotation." + new)
                legacy = importlib.import_module("scripts." + old)
                self.assertIs(legacy, canonical)
        script = (
            "import importlib; mapping=" + repr(MAPPING) + "; "
            "[(lambda old,new: None if importlib.import_module('scripts.'+old) "
            "is importlib.import_module('emr_annotation.'+new) else "
            "(_ for _ in ()).throw(AssertionError(old)))(old,new) "
            "for old,new in mapping.items()]"
        )
        run = run_python("-c", script)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_legacy_private_symbols_and_patch_intercept_canonical_execution(self):
        canonical = importlib.import_module("emr_annotation.evaluation.entity_predictions")
        legacy_merge = importlib.import_module("scripts.merge_emr_data")
        canonical_merge = importlib.import_module("emr_annotation.data_preparation.merge_emr_data")
        self.assertIs(legacy_merge._clean, canonical_merge._clean)
        args = ["--reference", "missing.json", "--before", "missing-before.json",
                "--label-config", "missing.xml", "--output", "missing-result.json"]
        with patch("scripts.evaluate_entity_predictions.read_label_groups", side_effect=ValueError("compatibility sentinel")) as patched:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                canonical.main(args)
            patched.assert_called_once_with(Path("missing.xml"))

    def test_all_core_imports_work_without_site_packages_or_model_imports(self):
        modules = ["emr_annotation", "emr_annotation.annotation.schema", "emr_annotation.adjudication.workbench"]
        modules += ["emr_annotation." + new for new in MAPPING.values()]
        script = (
            "import importlib,sys,socket; "
            "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network')); "
            "[importlib.import_module(name) for name in " + repr(modules) + "]; "
            "assert not ({'torch','transformers','peft','model','config','requests'} & set(sys.modules)); "
            "assert not any(name.startswith('scripts.') for name in sys.modules)"
        )
        run = run_python("-c", script)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_all_existing_cli_help_options_match_canonical_module_entry(self):
        for old, new in MAPPING.items():
            module = importlib.import_module("emr_annotation." + new)
            if not hasattr(module, "main"):
                continue
            with self.subTest(module=old):
                old_run = run_python(ROOT / "scripts" / (old + ".py"), "--help")
                new_run = run_python("-m", "emr_annotation." + new, "--help")
                self.assertEqual(old_run.returncode, 0, old_run.stderr)
                self.assertEqual(new_run.returncode, 0, new_run.stderr)
                options = lambda value: set(re.findall(r"--[a-z][a-z-]*", value))
                self.assertEqual(options(old_run.stdout), options(new_run.stdout))

    def test_synthetic_entity_cli_outputs_identical_bytes_and_real_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            reference = folder / "reference.json"
            reference.write_text(json.dumps([
                {"task_id": "synthetic-1", "text": "发热", "entities": [{"start": 0, "end": 2, "label": "发热"}]},
                {"task_id": "synthetic-2", "text": "未见异常", "entities": []},
            ], ensure_ascii=False), encoding="utf-8")
            args = ["--reference", reference, "--before", reference, "--label-config",
                    ROOT / "label_studio/pneumonia_config.global-single.xml", "--dataset-kind", "synthetic_demo"]
            old_run = run_python(ROOT / "scripts/evaluate_entity_predictions.py", *args, "--output", folder / "old.json")
            new_run = run_python("-m", "emr_annotation.evaluation.entity_predictions", *args, "--output", folder / "new.json")
            self.assertEqual(old_run.returncode, 0, old_run.stderr)
            self.assertEqual(new_run.returncode, 0, new_run.stderr)
            self.assertEqual((folder / "old.json").read_bytes(), (folder / "new.json").read_bytes())
            result = json.loads((folder / "new.json").read_text("utf-8"))
            self.assertEqual(result["evaluator_sha256"], hashlib.sha256((ROOT / "emr_annotation/evaluation/entity_predictions.py").read_bytes()).hexdigest())

    def test_full_synthetic_training_conversion_keeps_old_new_data_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = ["--export", ROOT / "tests/fixtures/label_studio_demo_export.json", "--selection",
                    ROOT / "tests/fixtures/label_studio_demo_selection.json", "--label-config",
                    ROOT / "label_studio/pneumonia_config.global-single.xml", "--ratios", "0.6", "0.2", "0.2",
                    "--dataset-kind", "synthetic_demo"]
            old_run = run_python(ROOT / "scripts/prepare_training_data.py", *args, "--output-dir", folder / "old")
            new_run = run_python("-m", "emr_annotation.training_data.preparation", *args, "--output-dir", folder / "new")
            self.assertEqual(old_run.returncode, 0, old_run.stderr)
            self.assertEqual(new_run.returncode, 0, new_run.stderr)
            old_files = sorted(p.relative_to(folder / "old") for p in (folder / "old").rglob("*") if p.is_file())
            new_files = sorted(p.relative_to(folder / "new") for p in (folder / "new").rglob("*") if p.is_file())
            self.assertEqual(old_files, new_files)
            for relative in old_files:
                with self.subTest(file=str(relative)):
                    old = json.loads((folder / "old" / relative).read_text("utf-8"))
                    new = json.loads((folder / "new" / relative).read_text("utf-8"))
                    if isinstance(old, dict):
                        old.pop("generated_at_utc", None)
                        new.pop("generated_at_utc", None)
                    self.assertEqual(old, new)

    def test_review_package_resolves_docs_and_rewrites_any_report_link_depth(self):
        from emr_annotation.adjudication.package_review import package
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            review = folder / "arbitrary" / "depth" / "review"
            review.mkdir(parents=True)
            (review / "review.html").write_text("<!DOCTYPE html><p>Synthetic review only</p>", "utf-8")
            (review / "核查报告.md").write_text("[方案](../../../../../../documentation/double_annotation_adjudication/double-annotation-adjudication.md)", "utf-8")
            for filename in ["裁决模板.json", "标签核查签署模板.json", "验证记录_v2.json"]:
                (review / filename).write_text('{"synthetic": true}', "utf-8")
            result = package(review, folder / "package")
            self.assertTrue(result["page_copies_byte_identical"])
            self.assertEqual((folder / "package/operator/核查报告.md").read_text("utf-8"), "[方案](裁决方案.md)")
            self.assertEqual((folder / "package/operator/裁决方案.md").read_bytes(), (ROOT / "documentation/double_annotation_adjudication/double-annotation-adjudication.md").read_bytes())

    def test_synthetic_double_annotation_cli_keeps_payload_and_template_at_any_output_location(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "synthetic.json"
            tasks = json.loads((ROOT / "tests/fixtures/label_studio_demo_export.json").read_text("utf-8"))
            for task in tasks:
                task["project"] = "synthetic-project"
                task["data"]["task_id"] = "synthetic-" + str(task["id"])
            source.write_text(json.dumps(tasks, ensure_ascii=False), "utf-8")
            source_bytes = source.read_bytes()
            args = ["--input", source, "--project-config", "synthetic-project=" + str(ROOT / "label_studio/pneumonia_config.global-single.xml")]
            old_run = run_python(ROOT / "scripts/audit_double_annotations.py", *args, "--output", folder / "old")
            new_run = run_python("-m", "emr_annotation.annotation_analysis.double_annotations", *args, "--output", folder / "new")
            self.assertEqual(old_run.returncode, 0, old_run.stderr)
            self.assertEqual(new_run.returncode, 0, new_run.stderr)
            self.assertEqual(source.read_bytes(), source_bytes)
            for filename in ["audit.json", "review.html", "裁决模板.json", "标签核查签署模板.json"]:
                self.assertEqual((folder / "old" / filename).read_bytes(), (folder / "new" / filename).read_bytes())
            report = (folder / "new/核查报告.md").read_text("utf-8")
            self.assertIn("不可评价", report)
            target = re.search(r"\[可复用裁决方案\]\(([^)]+)\)", report).group(1)
            if target.startswith("file:"):
                self.assertEqual(target, (ROOT / "documentation/double_annotation_adjudication/double-annotation-adjudication.md").as_uri())
            else:
                self.assertEqual((folder / "new" / target).resolve(), (ROOT / "documentation/double_annotation_adjudication/double-annotation-adjudication.md").resolve())

    def test_double_annotation_cli_without_case_choices_finishes_all_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "synthetic-empty-annotations.json"
            source.write_text(json.dumps([{
                "id": 1, "project": "synthetic-empty", "data": {"text": "虚构病例", "task_id": "fictional-1"},
                "annotations": [{"id": 11, "completed_by": "fictional-A", "result": []},
                                {"id": 12, "completed_by": "fictional-B", "result": []}],
            }], ensure_ascii=False), "utf-8")
            source_bytes = source.read_bytes()
            args = ["--input", source, "--project-config", "synthetic-empty=" + str(ROOT / "label_studio/pneumonia_config.global-single.xml")]
            for route, entry in [("old", [ROOT / "scripts/audit_double_annotations.py"]),
                                 ("new", ["-m", "emr_annotation.annotation_analysis.double_annotations"])]:
                with self.subTest(route=route):
                    run = run_python(*entry, *args, "--output", folder / route)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(sorted(p.name for p in (folder / route).iterdir()),
                                     sorted(["audit.json", "review.html", "核查报告.md", "裁决模板.json", "标签核查签署模板.json"]))
                    audit = json.loads((folder / route / "audit.json").read_text("utf-8"))
                    self.assertIsNone(audit["summary"]["case"])
                    self.assertIsNone(audit["summary"]["projects"]["synthetic-empty"]["case"])
                    self.assertIsNone(audit["summary"]["projects"]["synthetic-empty"]["entity"]["symmetric_f1"])
                    self.assertIn("病例总体一致：不可评价（无可比病例）", (folder / route / "核查报告.md").read_text("utf-8"))
                    template = json.loads((folder / route / "裁决模板.json").read_text("utf-8"))
                    self.assertTrue(all(item["status"] == "待裁决" for item in template["decisions"]))
                    self.assertNotIn("__PAYLOAD__", (folder / route / "review.html").read_text("utf-8"))
            self.assertEqual(source.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
