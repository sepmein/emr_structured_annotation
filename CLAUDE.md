# CLAUDE.md

data 顶层名称和版本按[数据目录索引](documentation/data-directory-naming.md)：6 个目录已改名为用途/日期/`v001`，旧名称为本地隐藏 Junction；新任务引用新路径，遍历跳过 Junction。内部冻结材料、根 CSV、旧清单内容保持；`v001` 表示首次登记的存储快照。

## 每次新任务先读

读取根目录 [AGENTS.md](AGENTS.md) 和 [项目组织方案与规则](documentation/project-organization.md)。代码按工作流程分块；维护文档、外部输入和运行产物按该方案分别归位。新任务以 AGENTS.md 的当前项目约束为准；下文概述若与当前规则不同，先核对现有实现，不套用旧说明。

## Project Overview

当前文档入口：[文档总导航](documentation/README.md)；文档移动、兼容入口及内容哈希见[迁移记录](documentation/migration-log.md)。旧文档路径为导航页，维护正文位于对应 stage 目录。
阶段 2 共享业务位置和兼容入口见[源码移动快照](documentation/source-migration.json)；阶段 3 四个模型实现模块及旧别名见[模型迁移快照](documentation/model-migration.json)。新业务修改进入对应模块，保留现有命令；73 件明确材料的副本归位与保留索引已完成，见[旧资料索引](documentation/legacy-materials.md)；原件与兼容目录保留。阶段 5 已完成本轮范围整体验证：当前 144 项离线测试及净源码沙箱 144 项通过，实际模型/服务验收仍未完成。后续目录改名映射与命名规则见[目录命名规范](documentation/directory-naming.md)；当前服务在 `label_studio_backend/service.py`，`backend.service` 仅为兼容别名，标注技能源在 `annotation_skills/`。

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

ModernBERT retains script-style top-level `model`, `config`, `train_ner_re`, `modeling`, `label_studio_backend`, and `training` imports; `backend.service` remains a compatibility alias. Do not switch the service to `python -m modernbert_ml_backend._wsgi` or mix package-qualified/top-level config modules without adapting and validating this namespace. The script entry still loads the local `label_studio_ml/` runtime; preserve it and the independent requirements environment.

## Architecture

- `emr_annotation/data_preparation/merge_emr_data.py`: stdlib multi-table merge keyed by `patient_id + serial_number`, combined text, age grouping, and Label Studio task export. `scripts/merge_emr_data.py` is its compatibility CLI; the merge is not deidentification.
- `emr_annotation/annotation/schema.py`: shared readers with distinct export-analysis and double-review data contracts.
- `emr_annotation/annotation_analysis/`: export auditing, reviewed-reference selection and descriptive double-annotation comparison.
- `emr_annotation/adjudication/`: rendering and review packaging; the maintained template is `emr_annotation/adjudication/templates/double_annotation_review.html`.
- `emr_annotation/training_data/`: conversion, patient grouping, frozen-data validation and token alignment checks.
- `emr_annotation/evaluation/`: entity/NER+RE scoring, regex baseline, service clients and benchmark comparison; no model loading or requests on import.
- `scripts/`: 14 compatibility CLI/import aliases, plus separate WHO collection and synthetic training-smoke tools. New shared logic belongs in the canonical business modules.
- `label_studio/pneumonia_config.xml`: current shared adult/child schema. The Text control is named `chief_complaint_text` and reads `$text` from task `data.text`; offsets must refer to the exact same text.
- `modernbert_ml_backend/modeling/network.py`: joint network; `modeling/inference.py`: text decoding and predictor.
- `modernbert_ml_backend/label_studio_backend/service.py`: Label Studio adapter, Schema and fit trigger; `model.py` is its compatibility module alias.
- `modernbert_ml_backend/training/trainer.py`: datasets, training, evaluation and saving, directly depending on the network; `train_ner_re.py` preserves the old alias and CLI. Training does not import the service or SDK.
- `modernbert_ml_backend/config.py`: local model paths, dynamic label mappings, tokenizer, and training settings. Base models live in `pretrained_models/`; trained outputs in `output/{base_model}/{schema}/best_model/`.
- `annotation_agent_workflow/`: file-based pilot preparation, privacy audits, result validation, comparison, adjudication, and historical round records. Do not rewrite historical results.
- `documentation/annotation/`: current Markdown guide/dictionary, matching HTML reading pages, build scripts, and archived layout samples.
- `annotation_skills/`: medical annotator and guide developer skill sources.

The XML has 49 entity labels, 9 per-region attribute fields, 5 relations, and 1 case-level choice. Current ModernBERT code handles entities and relations; do not claim complete Choices support or verified multi-project state isolation. Preserve the legacy spelling `symptons_labels` in annotation contracts.

GLiNER was retired on 2026-10-08. `ml_backend/` Python sources and `main.py` were removed along with `gliner` / `gliner2` dependencies. Do not recreate that route from old round records. See `documentation/gliner-retirement.md` for the audit and recovery revision.

## Dependencies and validation boundaries

The root `pyproject.toml`/`uv.lock` and ModernBERT `requirements.txt` remain separate environments. Root Torch/Transformers/SentencePiece are retained for the existing notebook; this does not replace ModernBERT's pinned requirements. Never delete shared model packages, weights, data, or history as part of legacy-code cleanup.

`tests/` runs offline; `modernbert_ml_backend/predict_test.py` is manual integration/load testing against an actual service. Do not run it as an offline unit test. Static schema checks and data tests do not establish Label Studio UI compatibility, model performance, or human agreement.

阶段 3 根 `.venv` 的虚构 tiny forward 尝试因 PyTorch C 扩展导入失败而未执行模型；失败回执见模型快照。没有更改共享环境依赖，不将该尝试写为真实 forward、训练或服务验收通过。
