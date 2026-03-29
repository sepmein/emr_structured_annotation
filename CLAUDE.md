# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EMR Structured Annotation Framework — a Label Studio-based system for extracting structured information from Chinese Electronic Medical Records (EMR). The primary use case is pneumonia NER annotation using GLiNER2 as the ML backend.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the debug test script (loads model and runs a sample prediction)
uv run python -m ml_backend.test_predict

# Run the ml_backend server locally (development)
uv run python -m ml_backend._wsgi

# Run the gliner_backend via Docker
cd gliner_backend && docker compose up --build

# Run main.py (GLiNER2 smoke test)
uv run python main.py
```

## Architecture

There are two ML backend implementations — they serve the same purpose but differ in maturity:

### `ml_backend/` (active debug version)
- `model.py` — `PneumoniaNERModel(LabelStudioMLBase)`: lazy-loads GLiNER2, builds a schema from `GLINER2_LABELS`, runs extraction, then applies a **re-anchoring fix** to correct character offset drift between GLiNER2 output and the rendered Label Studio text.
- `prompts.py` — single source of truth for all NER labels. `LABEL_PROMPTS` maps Chinese label names → `(GLiNER2 prompt string, from_name)`. `GLINER2_LABELS` is derived from it. All labels use a flat threshold (`GLINER_THRESHOLD` env var, default `0.4`).
- `_wsgi.py` — Flask app entry point via `label_studio_ml.api.init_app`.

### `gliner_backend/` (more complete, Docker-deployed version)
- `model.py` — `GLiNERModel(LabelStudioMLBase)`: same concept but uses `LABEL_SCHEMA` with **per-entity thresholds** (assertion labels use 0.55–0.6, symptom labels use 0.35–0.4). Supports fine-tuned model loading with fallback to pretrained. Implements `fit()` for training trigger from Label Studio.
- Deployed via `docker-compose.yml` on port `9091`.

### `label_studio/`
- XML labeling configs for Label Studio projects. `pneumonia.xml` is the current production config.
- The canonical text field is `chief_complaint_text`, which renders up to 7 EMR visits from `emr_activity_info[0..6]` (each with `activity_time`, `chief_complaint`, `present_illness_his`).
- **Critical**: The whitespace/indentation in the XML `<Text value="...">` template must exactly match what `_extract_text()` in `ml_backend/model.py` reconstructs — any mismatch causes annotation offset errors.

## Label Schema

Labels are organized into 8 `from_name` groups, all targeting `chief_complaint_text`:
- `symptons_labels` — clinical symptoms (发热, 气促, etc.)
- `diagnosis_labels` — imaging/pathology findings
- `treatment_labels` — interventions (机械通气)
- `epidemics_labels` — epidemiological exposure history
- `time_labels` — temporal modifiers (当前/既往/持续/进行性加重)
- `status_labels` — assertion modifiers (肯定/否定/可疑/条件性/假设性/与患者本人无关)
- `measure_entities` — measurement indicator names (呼吸频率, SpO2, etc.)
- `measure_labels` — measurement attributes (数值/单位/比较符/阈值判断)
- `pathogen` / `bacteria` / `other_pathogen` — pathogen entities

Relations defined in `pneumonia.xml`: 状态, 时间, 对应指标, 提示病原, 改善对象.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GLINER_MODEL` | `fastino/gliner2-base-v1` | Model checkpoint (ml_backend) |
| `GLINER_MODEL_NAME` | `fastino/gliner2-base-v1` | Model checkpoint (gliner_backend) |
| `GLINER_THRESHOLD` | `0.4` | Global confidence threshold (ml_backend) |
| `THRESHOLD` | `0.4` | Default threshold (gliner_backend) |
| `MODEL_DIR` | `./models` / `/data/models` | Directory for fine-tuned model |
| `FINETUNED_MODEL_PATH` | `finetuned_model` | Subdirectory within MODEL_DIR |
| `LABEL_STUDIO_URL` | `http://localhost:8080` | Label Studio instance URL |
| `LABEL_STUDIO_API_KEY` | — | Required for training trigger |
| `PORT` | `9090` | ML backend server port |

## Key Design Decisions

- **Re-anchoring**: `ml_backend/model.py` searches for entity text near the model-predicted offset to correct drift. This is necessary because GLiNER2 may shift character positions when processing long concatenated texts.
- **`gliner_backend` is the more production-ready implementation** — it uses per-entity thresholds, proper `ModelResponse`/`PredictionValue` types, and Docker deployment. `ml_backend` is a debug/development variant.
- The `pathogen`/`bacteria`/`other_pathogen` label groups use `toName="text"` in `pneumonia.xml` (not `chief_complaint_text`) — this is a known inconsistency in the XML config.
