# AGENTS.md

## 每次新任务：组织规则与执行顺序

本文件是仓库任务规则入口。每次新任务先阅读本文件及
[项目组织方案与规则](documentation/project-organization.md)，再阅读涉及模块的当前说明。
后者包含按工作流程划分的代码、文档、数据、输出位置和分阶段迁移方案。
源码分层及 stage 文档目录已落地，明确旧材料的副本归位与索引已完成；以当前映射核对入口，不把未实施的建议路径当作现有入口。
当前文档统一导航见 [documentation/README.md](documentation/README.md)，已完成的移动及哈希见
[迁移记录](documentation/migration-log.md)。顶层旧文档页仅保留导航，维护和打包使用 stage 目录中的正文。
阶段 2 共享业务已迁至 `emr_annotation/`；[源码移动快照](documentation/source-migration.json)记录原实现、兼容入口及完成时哈希。`scripts/` 的 14 个旧业务入口继续可用，新增共享逻辑须进入相应业务包。阶段 3 模型源码拆分及离线验证已完成，见[模型迁移快照](documentation/model-migration.json)；指定环境真实模型/服务验收仍未完成。阶段 4 已完成 73 件明确材料的副本归位与保留索引，见[旧资料索引](documentation/legacy-materials.md)；阶段 5 已完成本轮源码、文档和明确材料范围的整体验证：当前 144 项离线测试及净源码沙箱 144 项通过。旧 tmp、冻结记录、原始导出和模型目录的原位保留是兼容例外，不代表所有旧目录已物理迁移。

1. 先检查工作区已有改动，找到现有实现、调用者、命令入口和相关测试；保留用户正在进行的工作。
2. 明确任务属于数据准备、标注规范、模型构建、Label Studio 后端、标注分析、双人裁决、训练数据冻结、训练、评估或交付中的哪一块，再确定文件归属。
3. 新功能先复用已有转换、Schema 解析和评分逻辑。可复用业务逻辑与命令入口分离；不要继续让 `scripts/` 成为共享实现的无限堆积处。新包入口不得自动导入模型库、联网或加载权重。
4. 四类内容分开：源码/模板留在代码区；维护说明放 `documentation/`；外部原始数据与导出放 `data/`；派生数据、分析报告、工作台、训练和评估产物放 `output/`。禁止将真实病历、凭据或运行结果混入源码、测试夹具或文档示例。
5. 新批次使用 `data/<batch_id>/...` 和 `output/<batch_id>/<stage>/<run_id>/...`。同批次各阶段使用相同 batch_id；新运行使用新 run_id，不覆盖旧结果。记录输入路径/哈希、项目 Schema 快照、代码版本及未提交状态、参数、阶段、输出和验证情况；不得伪称未实施的校验已完成。
6. 基础权重统一放在 `pretrained_models/<base_model>/`，旧 `bert-base-model/` 为本地隐藏 Junction 兼容入口，跨工作区恢复须重建并核对哈希。训练权重 `output/{base_model}/{schema}/best_model/` 保留原路径，不随基础权重根改名。已冻结的数据、历史 manifests、轮次记录和版本证据保持原位原内容，通过索引关联。
7. `tmp/` 仅供临时工作，不得成为正式入口、长期数据源或唯一交付位置；可复用工具应迁入对应代码模块，可复现产物归入 output。清理已有已跟踪文件前逐项判断并保留必要证据，禁止整目录删除。
8. 当前指南/字典维护源为 `documentation/annotation/` 的 Markdown；指南 HTML 由现有构建脚本生成，字典 HTML 目前需单独同步核对。`annotation_agent_workflow/guides/` 与 `runs/` 为历史依据，不能当成当前默认版本。
9. 移动文件前检查导入、文档命令、模板路径、相对模型路径、测试、打包脚本和来源哈希；保留现有 CLI 兼容入口。一次迁移一块，验证后再进入下一块。不得顺带恢复 GLiNER、统一两套环境或删除本地 ML runtime。
10. 修改组织结构时同步更新当前 README、模块文档和本方案中的迁移状态。按改动运行相关离线检查；模型/服务/医学验证单独报告。收尾说明实际改变、验证和未完成阶段。
11. 目录命名参照 [目录命名规范](documentation/directory-naming.md)：采用英文小写和下划线，名称体现内容用途。当前文档目录为 `annotation`、`training_data_preparation`、`model_training`、`model_evaluation`、`double_annotation_adjudication`、`project_delivery`；技能源为 `annotation_skills/`，服务实现为 `modernbert_ml_backend/label_studio_backend/`。旧 `backend/` 仅保留导入兼容，新代码使用 `label_studio_backend`。运行清单的 stage 标识保持原规则，不随文档目录改名；禁止为统一名称改写历史路径或模型缓存路径。
12. data 顶层目录按[数据目录用途、批次与版本规则](documentation/data-directory-naming.md)使用 `<用途>[_范围][_批次]_<YYYYMMDD>_vNNN`。本次用户指定的 6 个顶层改名保留旧路径为隐藏 Windows Junction，原文件内容及内部冻结名称保持；新任务使用新目录，清点时跳过 Junction 避免重复。`v001` 是首次登记的存储快照，不是标注质量/UI/模型版本；后续输入修订新建版本目录，处理重跑新增 output run_id。根目录 16 份 CSV 及其它冻结/历史位置继续保留；换工作区或备份恢复时按映射核对本地兼容入口。

规则优先级：用户本次明确指令优先；本文件定义项目约束；组织方案提供映射与步骤；历史记录只用于追溯。发现冲突先核对当前代码，不照搬旧文档。

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

ModernBERT retains script-style top-level `model`, `config`, `train_ner_re`, `modeling`, `label_studio_backend`, and `training` imports; `backend.service` remains a compatibility alias. Do not switch the service to `python -m modernbert_ml_backend._wsgi` or mix package-qualified and top-level config imports without adapting and validating the namespace. Its script entry loads the local `modernbert_ml_backend/label_studio_ml/` runtime; do not remove that runtime as duplicate code without checking local changes.

## Architecture

- `emr_annotation/data_preparation/merge_emr_data.py`: stdlib multi-table merge keyed by `patient_id + serial_number`, combined text, age grouping, and Label Studio task export. `scripts/merge_emr_data.py` remains its compatibility CLI; this is not a deidentification tool.
- `emr_annotation/annotation/schema.py`: shared Schema readers; export-analysis and double-review readers intentionally return distinct contracts. Preserve descendant Label parsing, choice aliases and legacy control names.
- `emr_annotation/annotation_analysis/`: export validation, reference selection and descriptive double-annotation comparison; no automatic medical adjudication.
- `emr_annotation/adjudication/`: workbench rendering and review packaging; maintained HTML source is `templates/double_annotation_review.html` inside this package.
- `emr_annotation/training_data/`: training conversion, patient grouping, frozen-data checks and tokenizer alignment audit.
- `emr_annotation/evaluation/`: strict entity/NER+RE scoring, regex baseline, model/chat service clients and benchmark comparison. Importing these modules does not load models or send requests.
- `scripts/`: compatibility CLI/import aliases for relocated business logic; WHO collection and synthetic training-smoke setup remain separate tools. Edit the canonical business module, not a second implementation in an alias.
- `label_studio/pneumonia_config.xml`: current shared adult/child schema. The Text control is named `chief_complaint_text` and reads `$text` from task `data.text`; offsets must refer to the exact same text.
- `modernbert_ml_backend/modeling/network.py`: joint NER+RE network; `modeling/inference.py`: text decoding and `NERREPredictor`, shared by consumers.
- `modernbert_ml_backend/label_studio_backend/service.py`: Label Studio Schema synchronization, prediction conversion and delayed fit trigger. `model.py` is the legacy module alias, not a second network implementation.
- `modernbert_ml_backend/training/trainer.py`: dataset conversion, training, evaluation and saving; it imports the network directly and does not import the service or Label Studio SDK. `train_ner_re.py` retains its alias and CLI.
- `modernbert_ml_backend/config.py`: local model paths, dynamic label mappings, tokenizer, and training settings. Base models live in `pretrained_models/`; trained outputs in `output/{base_model}/{schema}/best_model/`.
- `annotation_agent_workflow/`: file-based pilot preparation, privacy audits, result validation, comparison, adjudication, and historical round records. Do not rewrite historical results.
- `documentation/annotation/`: current Markdown guide/dictionary, matching HTML reading pages, build scripts, and archived layout samples.
- `annotation_skills/`: medical annotator and guide developer skill sources.

The XML has 49 entity labels, 9 per-region attribute fields, 5 relations, and 1 case-level choice. Current ModernBERT code handles entities and relations; do not claim complete Choices support or verified multi-project state isolation. Preserve the legacy spelling `symptons_labels` in annotation contracts.

The default XML preserves the 9 legacy entity control names, with `choice="single"` within each group, for projects with existing annotations. `label_studio/pneumonia_config.global-single.xml` puts all 49 labels in one `symptons_labels` control for new or explicitly migrated projects; nested Views there are visual groups only. Parse descendant Label nodes, not just direct children. Match exports to the configuration used by their project; see `documentation/annotation/label-studio-single-selection.md`. The synthetic demo uses the global-single variant. Do not rewrite historical exports or frozen manifests to match the current UI.

GLiNER was retired on 2026-10-08. `ml_backend/` Python sources and `main.py` were removed along with `gliner` / `gliner2` dependencies. Do not recreate that route from old round records. See `documentation/gliner-retirement.md` for the audit and recovery revision.

## Dependencies and validation boundaries

The root `pyproject.toml`/`uv.lock` and ModernBERT `requirements.txt` remain separate environments. Root Torch/Transformers/SentencePiece are retained for the existing notebook; this does not replace ModernBERT's pinned requirements. Never delete shared model packages, weights, data, or history as part of legacy-code cleanup.

`tests/` runs offline; `modernbert_ml_backend/predict_test.py` is manual integration/load testing against an actual service. Do not run it as an offline unit test. Static schema checks and data tests do not establish Label Studio UI compatibility, model performance, or human agreement.

阶段 3 在根 `.venv` 尝试虚构 tiny forward 时，PyTorch C 扩展导入失败；未执行 forward、解码、训练或服务。失败回执位置见模型迁移快照。保留两套环境和已有依赖，不将离线/模拟通过写为指定 requirements 环境已验收。
