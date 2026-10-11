# 项目文档总导航

新任务先阅读根目录 [AGENTS.md](../AGENTS.md) 和 [项目组织方案与规则](project-organization.md)。阶段 1–3 文档归位、共享业务及模型源码拆分/离线验证已完成；阶段 4 已完成 73 件明确材料的副本归位与保留索引，阶段 5 已完成本轮源码、文档和明确材料范围整体验证（当前离线 144 项，净源码沙箱 144 项）。本页维护当前流程入口，历史证据保留在原位置。实际模型/服务验收仍未完成。

本机环境修复、独立 GPU 环境及后续检查命令见[本地训练与评估环境](model_training/runtime-environment.md)。新的实际运行验收另行记录，不改写历史迁移回执。

基座替换试验见[mmBERT与中文BERT对比](model_training/mmbert-base-comparison.md)，复用冻结AI样例与现有多任务网络；实际结果和验证凭据保存在对应output运行中。

人工审查完成后的流程见[mmBERT混合判断训练](model_training/human-reviewed-hybrid.md)：人工参考冻结、1000条规则样本、预测实体/关系与文本融合，保留真实患者分组及辅助标签来源边界。

进一步改进见[mmBERT优化与小模型分类对照](model_training/mmbert-optimization.md)，属性头、全检测特征缺口及文献建议见[Laya架构分析](model_training/laya-architecture-analysis.md)。

当前执行目标已补充为[逐项错分原因分析与模型改进](model_training/error-driven-improvement.md)，保留性能验收目标、人工参考和真实患者评估边界。

人工审查后质量统计见[双标一致性与裁决符合度](double_annotation_adjudication/human-review-statistics.md)：分别报告双人相互一致、各人与参考相符以及可追溯的差异案例。

阶段成果汇报见[项目阶段汇报 PPT 维护入口](project_delivery/project-progress-presentation.md)：由复核快照生成可编辑图表，区分已完成工作、实测结果与尚未达到的目标。

## 按工作流程查找

目录含义、改名映射及后续命名约定见[目录命名规范](directory-naming.md)；本轮 7 个目录已完成改名，原阶段快照仍记录当时路径。

本地 data 的 6 个目录已按用途、日期和 `v001` 存储版本命名，映射、兼容入口及后续版本规则见[数据目录索引](data-directory-naming.md)。

| 模块 | 当前说明 |
|---|---|
| 数据准备 | [现有合并入口与数据契约](../README.md#emr-数据拼合)；[历史试标工作流](../annotation_agent_workflow/RUNBOOK.md) |
| 标注规范 | [指南、字典与阅读页面](annotation/README.md)；[实体单选配置与旧项目兼容](annotation/label-studio-single-selection.md) |
| 模型构建与 Label Studio 后端 | [当前实现和服务边界](../README.md#3-modernbert-后端的边界)；[网络](../modernbert_ml_backend/modeling/network.py)、[推理解码](../modernbert_ml_backend/modeling/inference.py)、[服务适配](../modernbert_ml_backend/label_studio_backend/service.py)；旧 model.py 为兼容别名 |
| 标注结果分析 | [导出核查与参考准备](training_data_preparation/label-studio-evaluation-preparation.md)；[双标审计与复核](double_annotation_adjudication/double-annotation-adjudication.md) |
| 双人裁决 | [裁决方案](double_annotation_adjudication/double-annotation-adjudication.md)；[可视化工作台](double_annotation_adjudication/double-annotation-workbench.md)；[多项目整合与统计流程](double_annotation_adjudication/trial-batch-workflow.md)；[全量人工审查与裁决工具](double_annotation_adjudication/human-review-tool.md) |
| 训练数据准备与冻结 | [完整导出到训练格式](training_data_preparation/label-studio-to-training.md)；[参考选择与患者分组](training_data_preparation/label-studio-evaluation-preparation.md)；[冻结清单核查与训练前提](training_data_preparation/frozen-split-training.md) |
| 模型训练 | [裁决后训练与训练前后评估](model_training/adjudicated-model-training.md)；[冻结清单独立入口](training_data_preparation/frozen-split-training.md)；[实际训练模块](../modernbert_ml_backend/training/trainer.py) |
| 模型评估 | [实体逐标签评价](model_evaluation/model-evaluation.md)；[正则基线](model_evaluation/regex-baseline-comparison.md)；[服务预测与测速](model_evaluation/model-service-benchmark.md)；[通用大模型测试](model_evaluation/chat-model-benchmark.md)；[长度、速度与容量情景](model_evaluation/model-speed-capacity-comparison.md)；[200条真人双标实施方案](model_evaluation/200条真人双标_训练评估实施方案.md) |
| 资料构建与交付 | [工作台发布文件选择](project_delivery/double-annotation-release.md)；[汇报讨论底稿](project_delivery/国家疾控局来访汇报_讨论底稿.md)；[讲述提纲](project_delivery/国家疾控局来访汇报_讲述提纲_v0.1.md)；[待补事项与定稿条件](project_delivery/国家疾控局来访汇报_待补事项与定稿条件.md) |
| 辅助研究 | [WHO材料采集入口](../scripts/fetch_who_don_pdf.py)；[探索 notebook](../Untitled.ipynb) |

## 内容存放边界

维护说明在下表对应的文档目录；当前指南套件继续位于 `annotation/`。文档目录使用更完整的用途名称，运行清单沿用原 stage 标识（对应关系见目录命名规范）。外部输入使用 `data/<batch_id>/`，新运行结果使用 `output/<batch_id>/<stage>/<run_id>/`。工具模板属于源码；生成的具体病例工作台属于运行结果。

技术文档顶层旧路径保留为导航页，不再维护第二份正文。发布/打包工具直接使用 stage 目录中的正文。旧链接带章节锚点时，请进入正文后定位章节。

## 当前共享业务实现

`emr_annotation/` 不在导入时加载模型、联网或生成文件。旧文档命令继续使用 `scripts/` 兼容入口；新消费者直接导入下表模块。完整原→新映射及阶段完成时哈希见[源码移动快照](source-migration.json)。

| 模块 | 实现位置与边界 |
|---|---|
| 数据准备 | [merge_emr_data.py](../emr_annotation/data_preparation/merge_emr_data.py)，合并不是脱敏 |
| Schema | [schema.py](../emr_annotation/annotation/schema.py)，`load_export_schema` 与 `load_double_annotation_schema` 保留不同结构，实体评分复用 `read_label_groups` |
| 标注分析 | [导出审计/参考选择](../emr_annotation/annotation_analysis/label_studio_evaluation.py)、[双标差异](../emr_annotation/annotation_analysis/double_annotations.py)，比较结果不自动成为医学裁决 |
| 裁决材料 | [工作台渲染](../emr_annotation/adjudication/workbench.py)、[HTML源码模板](../emr_annotation/adjudication/templates/double_annotation_review.html)、[打包](../emr_annotation/adjudication/package_review.py)，具体病例页面写入 output |
| 训练数据 | [转换](../emr_annotation/training_data/preparation.py)、[患者分组](../emr_annotation/training_data/split_reference.py)、[冻结预检](../emr_annotation/training_data/frozen.py)、[token对齐核查](../emr_annotation/training_data/token_audit.py) |
| 评价与测速 | [实体评价](../emr_annotation/evaluation/entity_predictions.py)、[NER+RE评价](../emr_annotation/evaluation/ner_re.py)、[正则基线](../emr_annotation/evaluation/regex_entity_baseline.py)、[服务客户端](../emr_annotation/evaluation/benchmark_model_service.py)、[大模型客户端](../emr_annotation/evaluation/benchmark_chat_model.py)、[基准比较](../emr_annotation/evaluation/compare_model_benchmarks.py) |

14 个旧脚本保留 CLI/导入兼容，模块别名保留同一实现对象、私有符号及 patch 行为；WHO 采集与训练 smoke 工具仍为独立辅助入口。

## 当前模型实现与命名空间

模型目录按 `modeling/network.py`、`modeling/inference.py`、`label_studio_backend/service.py`、`training/trainer.py` 分层。trainer 直接依赖 network，导入训练不加载服务或 Label Studio SDK；服务 fit 在训练输入就绪后延迟导入 trainer。两个旧文件 `model.py` 与 `train_ner_re.py` 为模块别名，后者同时保留 CLI。

运行仍采用 `python modernbert_ml_backend/_wsgi.py --port 9090`，以及原独立训练脚本。内部是顶层 `modeling` / `label_studio_backend` / `training` / `config` 命名空间，旧 `backend.service` 为同一实现的兼容别名，不代表已支持 `python -m modernbert_ml_backend._wsgi`。本地 `label_studio_ml/`、config、requirements、原模型路径保持。

实际范围见[模型迁移快照](model-migration.json)和[迁移记录](migration-log.md)。根 `.venv` 的 tiny 模型尝试因 PyTorch C 扩展导入失败而未执行 forward；`output/2026-10-10-code-migration/evaluation/model-separation-smoke-01/receipt.json`（本地 ignored 回执）保留。离线迁移通过不代表真实模型/服务、UI 或医学验收。

## 旧材料副本与保留索引

[旧项目资料保留与归位索引](legacy-materials.md)列出 73 件实际副本、保留目录、恢复方式及本地 ignored 清单限制。原件全部保留；其他 clone 不保证有 data/output 副本或未跟踪输入。

## 保留原位置的材料

- [已有虚构流程演示与冻结结果](evaluation_examples)：当前只建立导航，不搬迁，不改 JSON、manifest、报告或图表。
- [历史试标、裁决与轮次](../annotation_agent_workflow/runs)、[历史指南](../annotation_agent_workflow/guides)、[历史验证](../annotation_agent_workflow/validation)和[XML历史快照](../label_studio/archive)：只用于追溯，不作为新批次默认配置。
- [GLiNER退役审计](gliner-retirement.md)：保留审计与恢复路径，不恢复退役实现。
- [附件症状体征同义词提取](附件症状体征同义词提取.md)、[识别用语汇编](临床症状体征同义词及相关识别用语汇编_V1.0_2026-10-06.md)及顶层两份来源 PDF：来源 PDF 已建立 data 副本，原件和两份混合维护/提取 Markdown 继续保留；后者尚待逐段分类与消费者核对，见[旧资料保留索引](legacy-materials.md)。

具体移动与哈希见 [迁移记录](migration-log.md)。文件位置调整不构成医学、模型性能、服务或 UI 验收。

基础权重目录已统一为 `pretrained_models/`，旧 `bert-base-model/` 通过本地隐藏 Junction 兼容；规则及当前验证边界见[目录命名规范](directory-naming.md)。

中文多任务试验：[跨度实体、9属性和病例分类训练](model_training/schema-multitask-trial.md)，原始双标试验、患者分组、权重重载、六部分报告及关系未训练边界。

样例裁决试验：[AI逐点处理、患者边界冻结与训练报告](double_annotation_adjudication/ai-sample-adjudication.md)。AI样例与专家最终金标分开；未决点保留，关系正例监督及医学/UI验收边界单独列示。
