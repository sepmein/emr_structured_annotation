# 目录命名规范与当前映射

日期：2026-10-10。目录调整已落地；后续任务同时参照根目录 [AGENTS.md](../AGENTS.md) 和[项目组织方案](project-organization.md)。

## 本轮目录改名

| 原目录 | 当前目录 | 含义 |
|---|---|---|
| `skill_src/` | [annotation_skills/](../annotation_skills/) | 医学标注员与指南开发者的技能源 |
| `modernbert_ml_backend/backend/` | [modernbert_ml_backend/label_studio_backend/](../modernbert_ml_backend/label_studio_backend/) | Label Studio 服务适配、Schema 同步和训练触发 |
| `documentation/training_data/` | [training_data_preparation/](training_data_preparation/) | 标注导出到训练格式、患者分组及数据冻结说明 |
| `documentation/training/` | [model_training/](model_training/) | 模型训练及选模说明 |
| `documentation/evaluation/` | [model_evaluation/](model_evaluation/) | 模型评分、基线对照、服务测速及实验方案 |
| `documentation/adjudication/` | [double_annotation_adjudication/](double_annotation_adjudication/) | 双人标注差异裁决及工作台说明 |
| `documentation/delivery/` | [project_delivery/](project_delivery/) | 项目汇报、复核包发布和交付说明 |

`backend/` 留有两个兼容文件，`backend.service`、`model` 和 `label_studio_backend.service` 指向同一个实现对象；新代码只维护新目录中的 service。原 `scripts/` CLI、ModernBERT `_wsgi.py` 启动方式及顶层文档导航页继续可用。五个旧文档分组目录不保留第二份正文，当前导航和打包资源均使用新目录。

## 后续命名规则

1. 普通目录使用英文小写和下划线，名称体现用途，如 `model_evaluation`、`annotation_skills`。不新增 `new`、`misc`、`other`、`stuff` 等无法辨识职责的长期目录。
2. 不在 Python 包名前添加流程编号或中文。流程顺序写在导航和组织方案中，目录名不依赖当前流程排序。
3. 已有父目录提供清楚作用域时，子目录可保持简洁：`emr_annotation/evaluation` 是共享评价代码，`modernbert_ml_backend/training` 是模型训练代码；文档分组用完整用途名称，便于从文件列表直接辨认。
4. 四类根位置继续统一：源码在 `emr_annotation/`、`modernbert_ml_backend/` 等实现区；说明在 `documentation/`；外部输入在 `data/`；运行产物在 `output/`。不要另建平行的 `results/`、`reports/`、`datasets/` 根目录。
5. `scripts/` 表示命令入口；`tests/fixtures/` 表示虚构测试输入；`annotation_skills/` 表示标注相关技能源；`label_studio/` 表示 XML 配置和配置历史。名称约定要写明内容边界。
6. `data/<batch_id>/`、`output/<batch_id>/<stage>/<run_id>/` 的 batch_id、stage、run_id 沿用组织方案。本次文档目录改名不修改运行阶段标识或旧 receipt/manifest。
7. 改名前检查导入、CLI、文档链接、模板、生成脚本、打包、代码哈希记录及测试。必要时保留薄兼容入口，禁止同时维护两份业务实现。
8. 历史轮次、冻结结果、模型权重路径、来源快照和工具环境目录保留原名称。命名规则用于新目录和经验证的当前维护目录，不能直接覆盖历史来源。用户指定的 data 顶层目录改名按[数据目录规则](data-directory-naming.md)执行：内容和内部冻结名称保持，旧根路径通过隐藏 Junction 兼容访问。

## 文档目录与运行阶段的对应

| 运行清单 stage（保持原值） | 当前维护文档目录 |
|---|---|
| `annotation` | `documentation/annotation/` |
| `training_data` | `documentation/training_data_preparation/` |
| `training` | `documentation/model_training/` |
| `evaluation` | `documentation/model_evaluation/` |
| `adjudication` | `documentation/double_annotation_adjudication/` |
| `delivery` | `documentation/project_delivery/` |

数据准备、模型构建、后端、标注分析及研究的当前入口继续从[总导航](README.md)查找；未来新增相应文档分组应采用 `data_preparation`、`modeling`、`label_studio_backend`、`annotation_analysis`、`research` 等明确名称。

## 本轮保留的兼容名称

`emr_annotation/` 和 `modernbert_ml_backend/` 已分别表明共享业务与模型运行环境，保留包根及模型脚本根。`label_studio/` 连同其 archive、`documentation/annotation/archive/`、`documentation/evaluation_examples/`、`annotation_agent_workflow/`、已有 output 批次和 `tmp/` 原件保持原位；data 顶层六个目录的新名称和旧路径兼容方式见数据目录规则。`.venv/` 等工具目录按工具约定使用。退役 `ml_backend/` 若只剩缓存，不构成当前实现入口。

完整改名前后路径/哈希、保护范围及实际验证见[目录调整记录](directory-renaming.json)。以前的源码/模型快照保留当时路径与哈希；[迁移记录](migration-log.md)追加本轮结果，不改写历史证据。

## 基础模型权重目录

`bert-base-model/` 已改为 `pretrained_models/`：名称表示基础模型与 tokenizer 权重，不限定 BERT 架构。代码默认使用新路径；本机旧路径保留隐藏 Windows Junction。两条路径指向相同文件，5 件现存虚构 tiny 冒烟模型文件的大小和 SHA-256 均未改变，详见 [权重目录改名记录](model-directory-renaming.json)。本次检查没有正式基础权重，不能视为真实模型已可运行。换工作区时将权重放到新目录，并按需要重建旧联接；权重及联接不进入 Git。

`modernbert_ml_backend/` 保留现名，内部 `modeling/`、`training/`、`label_studio_backend/` 分别组织网络、训练和服务；本地 `label_studio_ml/` runtime 保留。`ml_backend/` 本轮检查已不存在，GLiNER 路线继续退役。
