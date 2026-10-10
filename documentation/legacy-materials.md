# 旧项目资料保留与归位索引

日期：2026-10-10（Asia/Shanghai）。本页记录组织迁移阶段 4 已实际完成的范围。

后续 data 顶层目录已按用途/日期/存储版本改名，见[数据目录索引](data-directory-naming.md)。本页及原阶段清单中的 data 目标路径保留完成时名称，旧名称通过隐藏 Junction 仍访问同一文件；新任务使用索引中的新路径，不回写阶段 4 历史清单。

本轮采用**新目录副本归位、原件保留**：73 件已分类材料逐字节复制到统一的 data/output 批次目录；所有旧来源、用户已有修改、历史和现有模型路径保持原位。没有删除 tmp，没有整体取消其 Git 跟踪，也没有新增整目录忽略规则。

统一放置规则见`documentation/project-organization.md`及[AGENTS.md](../AGENTS.md)。本页用于历史资料来源定位，不代替当前模块入口。当前维护文档见`documentation/README.md`。

## 本次归位范围

本地完整清单：`legacy-project-migration` 批次，`delivery` 阶段，运行 `run_20261010_stage4_144604`；文件 `output/legacy-project-migration/delivery/run_20261010_stage4_144604/manifest.json`。清单含每项来源/目标、字节数、双方 SHA-256、原件 Git 跟踪状态、代码版本及开始时工作区状态，并保存保留索引和实际验证。该文件随 output 被 Git 忽略，不是受版本管理的永久副本；有需要时应与同批次材料一同备份。

下列 data/output 路径以及未跟踪输入只在持有本地材料的工作区可访问，其他 clone 可能缺失；这些位置用代码路径表示，受 Git 跟踪的来源保留导航链接。表中路径以仓库根目录为基准。

| 分类 | 件数 | 原位置 | 归位新位置 |
|---|---:|---|---|
| 指南 HTML/打印验证截图及 PDF | 13 | [tmp/](../tmp) 中下文逐项列出的图/PDF | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| 来源 PDF 页面图及文字提取 | 54（52 JPG、2 TXT） | [tmp/pdfs/](../tmp/pdfs) 的 JPG/TXT | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| WHO 2026-06-01 采集成果 | 3 | `output/who_disease_outbreak_news_index_2026-06-01.{html,json,pdf}` | `output/legacy-who-don-2026-06-01/research/run_20261010_stage4_144604/` |
| 外部来源 PDF | 2 | documentation 顶层下文两份 PDF | `data/legacy-source-documents/raw/run_20261010_stage4_144604/` |
| 外部人工筛选决定 | 1 | `tmp/emr_screening_decisions.json` | `data/emr_data_1_20261008/decisions/run_20261010_stage4_144604/` |

归位副本是既有材料的存储整理，未重新构建指南、重新采集 WHO、重新标注或重新生成人工决定。保留其旧文件名以方便查证，不能凭副本名称推断医学结论、公开发布许可或输入数据已去标识化。

## 保留索引与兼容例外

完整本地 metadata 索引记录 34,693 个已有文件（1,007,724,370 字节，其中 267 件为 Git 已跟踪文件）。索引只记录路径、大小、用途与是否 Git 跟踪，不解析病例或权重。统计范围是本轮开始时列举的 data、output、bert-base-model、tmp、workflow、历史示例/Schema、文档来源/派生稿、根目录 notebook 和原始导出；并非全仓库文件数量。归位后的 73 件副本列于 copies，不重复计入旧文件索引。

- [tmp/](../tmp)：86 件（79 tracked、7 untracked）全部留在原位。15 件 Python/CJS 一次性编辑、筛选、渲染或核查脚本仅建索引，本轮没有执行，也没有包装成可复用正式入口。其余旧 Markdown/XML 快照继续保留来源证据。后续复用前逐项核对依赖与导入副作用。
- `data/` 及根目录 `project-*-at-*.json`：原始 EMR、原始标注导出、人工签署和冻结输入保持原位；本轮不批量复制真实 EMR。根目录已有导出不得用改名/归位掩盖其 Git untracked 状态，后续经所有者确认来源与消费者后再逐批处理。
- `output/`：既有冻结集、训练/评估结果、人工复核材料和其他旧运行目录原位保留。本轮只另存已列出的 WHO 三件；既有 manifest/receipt 的旧来源路径与哈希保持不变。
- `bert-base-model/` 与 `output/{base_model}/{schema}/best_model/`：模型兼容路径继续保留，没有复制权重或切换模型消费者。
- [历史 workflow](../annotation_agent_workflow) 的 runs/guides/validation、`label_studio/archive/` 与[冻结虚构评估示例](evaluation_examples)：160 份既有文件通过原路径索引，不能重写成当前运行结果；本轮前后 SHA-256 全部一致。
- [指南历史布局](annotation/archive)、[研究 notebook](../Untitled.ipynb)：只记保留位置，不执行 notebook，也不以历史截图替代当前 Markdown/HTML 维护源。

## 来源 PDF 与派生稿为什么暂留旧位置

两份外部 PDF 已建立 `data/legacy-source-documents/raw/` 副本，旧 PDF 仍供现有引用和来源核对使用。页面图/直接文字提取副本归入 research 输出，其旧 tmp 原件全部保留。

[附件症状体征同义词提取](附件症状体征同义词提取.md)和[临床症状体征同义词及相关识别用语汇编](临床症状体征同义词及相关识别用语汇编_V1.0_2026-10-06.md)继续在 documentation 原位：当前尚未完成对人工维护段落、外部原文摘录及自动派生部分的逐段分类，也未验证其引用消费者。它们与纯 JPG/TXT 提取产物不同，不能仅根据文件名整体搬入 output；以后需单独确认维护正文与来源关系，再更新链接。

## 验证与恢复

- 73 个目标文件全部新建，目标路径位于仓库 data/output 中；分组文件数与预定清单一致。来源与目标字节数、SHA-256 均一致。
- 251 个保护来源（全部 86 件 tmp、160 件历史及其他本次来源去重）前后大小/哈希一致。原件未删除，原 tracked/untracked 状态保留；没有覆盖同名旧目标。
- 完成后独立复验全部 73 项复制与 251 个保护来源；当时本页 160 个本地文件/目录链接均存在（完成时快照；后续将 ignored/未跟踪位置改为代码路径），`git diff --check` 通过。此链接检查只针对本页，不证明其他 clone 拥有被忽略的本地资料。
- 本轮未读取病历字段、执行旧 tmp 脚本、重新标注、真实训练、模型/服务/UI 联调或医学验收。复制与静态检查不证明材料已可公开或模型有效。

恢复时先查本地 manifest 的 copies：若副本缺失，核对原件的 source_bytes/source_sha256 后，在新的 run_id 目录中逐项重新复制，再校验目标字节数和 SHA-256；不得覆盖已有目标。若原件缺失，只在确认原路径不存在后从已校验副本另存恢复。原件与副本都缺失时需找本地备份；普通 Git clone 不保证存在这些 ignored 资料。历史 frozen manifests 不改路径，继续按原始位置还原。无需通过回滚代码迁移取回本轮原件，因为原件全部保留。

## 以后新任务的目录例子

同批次阶段共享 batch_id，新运行使用未使用的 run_id。外部原件不可作为转换输出被覆盖；可复用工具进入对应业务模块，不能继续把正式工具放 tmp。

```text
data/emr_20261011/raw/                         # 新批次原始表
data/emr_20261011/label_studio_exports/        # 项目原始导出
data/emr_20261011/schema/                      # 对应项目 XML 原件/快照
data/emr_20261011/decisions/                   # 外部人工决定/签署
output/emr_20261011/annotation_analysis/run_20261011_01/
output/emr_20261011/adjudication/run_20261011_01/
output/emr_20261011/training_data/run_20261011_01/
output/emr_20261011/evaluation/run_20261011_01/
output/emr_20261011/delivery/run_20261011_01/
```

新指南截图或阅读 PDF 进入相应交付批次的 `output/<batch_id>/delivery/<run_id>/`；外部政策参考原件进入 `data/<batch_id>/raw/`，提取文字/页面图进入 `output/<batch_id>/research/<run_id>/`。保留每项输入路径/哈希和实际运行参数，避免把本次 legacy 归位批次当成新研究默认目录。

## 逐项来源与本地副本

下面全部 73 项来源指向原件；新运行目录列出本地副本位置。仅受 Git 跟踪的来源提供链接；ignored 目标、WHO 原件和未跟踪人工决定使用代码路径，其他 clone 不保证存在。

| 来源原件 | 本地新运行目录 |
|---|---|
| [tmp/case-decision-help-desktop.png](<../tmp/case-decision-help-desktop.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-cases.png](<../tmp/guide-html-cases.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-classification.png](<../tmp/guide-html-classification.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-desktop.png](<../tmp/guide-html-desktop.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-diagram-1.png](<../tmp/guide-html-diagram-1.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-diagram-4.png](<../tmp/guide-html-diagram-4.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-mobile-overflow.png](<../tmp/guide-html-mobile-overflow.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-mobile.png](<../tmp/guide-html-mobile.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-html-print.pdf](<../tmp/guide-html-print.pdf>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-print-page-26.png](<../tmp/guide-print-page-26.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-print-page-44.png](<../tmp/guide-print-page-44.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-print-page-6.png](<../tmp/guide-print-page-6.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/guide-print-verified.png](<../tmp/guide-print-verified.png>) | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_1.jpg](<../tmp/pdfs/0_1.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_10.jpg](<../tmp/pdfs/0_10.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_11.jpg](<../tmp/pdfs/0_11.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_12.jpg](<../tmp/pdfs/0_12.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_13.jpg](<../tmp/pdfs/0_13.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_2.jpg](<../tmp/pdfs/0_2.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_3.jpg](<../tmp/pdfs/0_3.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_4.jpg](<../tmp/pdfs/0_4.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_5.jpg](<../tmp/pdfs/0_5.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_6.jpg](<../tmp/pdfs/0_6.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_7.jpg](<../tmp/pdfs/0_7.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_8.jpg](<../tmp/pdfs/0_8.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/0_9.jpg](<../tmp/pdfs/0_9.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_1.jpg](<../tmp/pdfs/1_1.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_10.jpg](<../tmp/pdfs/1_10.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_11.jpg](<../tmp/pdfs/1_11.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_12.jpg](<../tmp/pdfs/1_12.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_2.jpg](<../tmp/pdfs/1_2.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_3.jpg](<../tmp/pdfs/1_3.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_4.jpg](<../tmp/pdfs/1_4.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_5.jpg](<../tmp/pdfs/1_5.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_6.jpg](<../tmp/pdfs/1_6.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_7.jpg](<../tmp/pdfs/1_7.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_8.jpg](<../tmp/pdfs/1_8.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/1_9.jpg](<../tmp/pdfs/1_9.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/overview.jpg](<../tmp/pdfs/overview.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-01.jpg](<../tmp/pdfs/render0-01.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-02.jpg](<../tmp/pdfs/render0-02.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-03.jpg](<../tmp/pdfs/render0-03.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-04.jpg](<../tmp/pdfs/render0-04.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-05.jpg](<../tmp/pdfs/render0-05.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-06.jpg](<../tmp/pdfs/render0-06.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-07.jpg](<../tmp/pdfs/render0-07.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-08.jpg](<../tmp/pdfs/render0-08.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-09.jpg](<../tmp/pdfs/render0-09.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-10.jpg](<../tmp/pdfs/render0-10.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-11.jpg](<../tmp/pdfs/render0-11.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-12.jpg](<../tmp/pdfs/render0-12.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render0-13.jpg](<../tmp/pdfs/render0-13.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-01.jpg](<../tmp/pdfs/render1-01.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-02.jpg](<../tmp/pdfs/render1-02.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-03.jpg](<../tmp/pdfs/render1-03.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-04.jpg](<../tmp/pdfs/render1-04.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-05.jpg](<../tmp/pdfs/render1-05.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-06.jpg](<../tmp/pdfs/render1-06.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-07.jpg](<../tmp/pdfs/render1-07.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-08.jpg](<../tmp/pdfs/render1-08.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-09.jpg](<../tmp/pdfs/render1-09.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-10.jpg](<../tmp/pdfs/render1-10.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-11.jpg](<../tmp/pdfs/render1-11.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/render1-12.jpg](<../tmp/pdfs/render1-12.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/renderoverview.jpg](<../tmp/pdfs/renderoverview.jpg>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/【9月28日】国家疾控局监测预警司关于征求《临床症候群识别规则(征求意见稿〉》意见的函.txt](<../tmp/pdfs/【9月28日】国家疾控局监测预警司关于征求《临床症候群识别规则(征求意见稿〉》意见的函.txt>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| [tmp/pdfs/关于印发《全国聚集性不明原因肺炎监测方案》的通知（国疾控综监测发〔2026〕22号）.txt](<../tmp/pdfs/关于印发《全国聚集性不明原因肺炎监测方案》的通知（国疾控综监测发〔2026〕22号）.txt>) | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| `output/who_disease_outbreak_news_index_2026-06-01.html` | `output/legacy-who-don-2026-06-01/research/run_20261010_stage4_144604/` |
| `output/who_disease_outbreak_news_index_2026-06-01.json` | `output/legacy-who-don-2026-06-01/research/run_20261010_stage4_144604/` |
| `output/who_disease_outbreak_news_index_2026-06-01.pdf` | `output/legacy-who-don-2026-06-01/research/run_20261010_stage4_144604/` |
| [documentation/【9月28日】国家疾控局监测预警司关于征求《临床症候群识别规则(征求意见稿〉》意见的函.pdf](<【9月28日】国家疾控局监测预警司关于征求《临床症候群识别规则(征求意见稿〉》意见的函.pdf>) | `data/legacy-source-documents/raw/run_20261010_stage4_144604/` |
| [documentation/关于印发《全国聚集性不明原因肺炎监测方案》的通知（国疾控综监测发〔2026〕22号）.pdf](<关于印发《全国聚集性不明原因肺炎监测方案》的通知（国疾控综监测发〔2026〕22号）.pdf>) | `data/legacy-source-documents/raw/run_20261010_stage4_144604/` |
| `tmp/emr_screening_decisions.json` | `data/emr_data_1_20261008/decisions/run_20261010_stage4_144604/` |

本页状态：阶段 4 的明确材料副本归位与保留索引已完成；原位置清理、一次性脚本复用审查、来源派生稿细分及未归位旧批次仍为后续工作，不能据此宣称所有旧文件已物理迁移。
