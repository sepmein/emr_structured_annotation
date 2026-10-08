# CLAUDE.md

## Project Overview

EMR Structured Annotation Framework: Chinese EMR preparation, Label Studio annotation, and ModernBERT NER + RE training/prediction. Outputs support reviewable public-health signals, not automatic clinical diagnosis.

## Commands

```sh
# Root environment and data preparation
uv sync
uv run python scripts/merge_emr_data.py

# Offline unit tests and documentation links; no model required
python -m unittest discover -s tests -v
python documentation/annotation/scripts/check-document-links.py

# ModernBERT: use an environment with modernbert_ml_backend/requirements.txt
python modernbert_ml_backend/_wsgi.py --port 9090
```

Do not switch ModernBERT to `python -m` without fixing its existing top-level `model`, `config`, and `train_ner_re` imports. Its script entry loads the local `modernbert_ml_backend/label_studio_ml/` runtime; do not remove that runtime as duplicate code without checking local changes.

## Architecture

- `scripts/merge_emr_data.py`: stdlib multi-table merge keyed by `patient_id + serial_number`, combined text, age grouping, and Label Studio task export. This is not a deidentification tool.
- `label_studio/pneumonia_config.xml`: current shared adult/child schema. The Text control is named `chief_complaint_text` and reads `$text` from task `data.text`; offsets must refer to the exact same text.
- `modernbert_ml_backend/model.py`: joint network, inference, schema parsing, prediction conversion, and training trigger.
- `modernbert_ml_backend/train_ner_re.py`: dataset conversion, training, evaluation, and saving.
- `modernbert_ml_backend/config.py`: local model paths, dynamic label mappings, tokenizer, and training settings. Base models live in `bert-base-model/`; trained outputs in `output/{base_model}/{schema}/best_model/`.
- `annotation_agent_workflow/`: file-based pilot preparation, privacy audits, result validation, comparison, adjudication, and historical round records. Do not rewrite historical results.
- `documentation/annotation/`: current Markdown guide/dictionary, matching HTML reading pages, build scripts, and archived layout samples.
- `skill_src/`: medical annotator and guide developer skill sources.

The XML has 49 entity labels, 9 per-region attribute fields, 5 relations, and 1 case-level choice. Current ModernBERT code handles entities and relations; do not claim complete Choices support or verified multi-project state isolation. Preserve the legacy spelling `symptons_labels` in annotation contracts.

GLiNER was retired on 2026-10-08. `ml_backend/` Python sources and `main.py` were removed along with `gliner` / `gliner2` dependencies. Do not recreate that route from old round records. See `documentation/gliner-retirement.md` for the audit and recovery revision.

## Dependencies and validation boundaries

The root `pyproject.toml`/`uv.lock` and ModernBERT `requirements.txt` remain separate environments. Root Torch/Transformers/SentencePiece are retained for the existing notebook; this does not replace ModernBERT's pinned requirements. Never delete shared model packages, weights, data, or history as part of legacy-code cleanup.

`tests/` runs offline; `modernbert_ml_backend/predict_test.py` is manual integration/load testing against an actual service. Do not run it as an offline unit test. Static schema checks and data tests do not establish Label Studio UI compatibility, model performance, or human agreement.
