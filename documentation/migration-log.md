# 项目迁移记录

日期：2026-10-10（Asia/Shanghai）。本记录针对本轮工作区的文档归位，不把先前用户修改记作本轮迁移。移动前哈希读取当时工作区的实际内容，不使用 Git HEAD 覆盖未提交工作。

## 阶段 1：当前文档归位

17 份维护文档按工作流程分组。原路径保留简短导航；新位置保存原正文，仅修正因目录层级变化而需要调整的本地链接。历史引用仍指向同一原文件，命令中的冻结数据/产物路径及历史数量、结论保持原文；没有修改历史文件。

README、AGENTS、CLAUDE、组织方案及 annotation 导航同步当前路径；打包脚本直接读取正文，并将新旧核查报告中的裁决方案链接转换为包内文件名。旧导航页不作为发布材料。

| 原路径 | 正文新路径 | 移动前 SHA-256 | 移动后正文 SHA-256 |
|---|---|---|---|
| `documentation/label-studio-to-training.md` | [documentation/training_data/label-studio-to-training.md](training_data_preparation/label-studio-to-training.md) | `7d4e659f3cf18403d2cc7a34f06e04a85465fe09b9584c9f6d3b36147e9091de` | `44bc2bf4b17928f008b7053f093803c6e9a7c91f608af49fe3ef2cca035384b6` |
| `documentation/label-studio-evaluation-preparation.md` | [documentation/training_data/label-studio-evaluation-preparation.md](training_data_preparation/label-studio-evaluation-preparation.md) | `8f3c8b6b2c323b9a7a0de165cc0497354cd112e6809180b3166d65194e68d0c8` | `b0efd62a310b332f1a6ef8acecf042307ad01df32bd2f90d62487f60dc579548` |
| `documentation/frozen-split-training.md` | [documentation/training_data/frozen-split-training.md](training_data_preparation/frozen-split-training.md) | `13ff14e54f81308ebb7f057a1b65d64a0dc173971f909cc17b0753e5e7b7572a` | `217afb5034da2dcc66fc76918f03a9ca5853a4fc6fa15b5c9814a46850308a27` |
| `documentation/adjudicated-model-training.md` | [documentation/training/adjudicated-model-training.md](model_training/adjudicated-model-training.md) | `c762be64199c44062a3979ce8703ddb5e42583b7859a25f26da719e0a28d74fc` | `71f045f3b423b01239ecfc3c8f31657ea851e421bb6cbe87c4a1e547fde68fdd` |
| `documentation/model-evaluation.md` | [documentation/evaluation/model-evaluation.md](model_evaluation/model-evaluation.md) | `f5a932a7709fe8c038ed3d721808da3aec442329cacac5ab25a06a67780581f4` | `95c373842ac4cb7c62f9cf6fdd33236099529bfe08692be9ca183579736af5ad` |
| `documentation/regex-baseline-comparison.md` | [documentation/evaluation/regex-baseline-comparison.md](model_evaluation/regex-baseline-comparison.md) | `003c86d2294b035d9743c954e590d4d6829e89602dedbdacb84e461116820849` | `83bea185fb2f1a5b55ea347cdef0a667a5d0617b526758a6266d663a6f777e45` |
| `documentation/model-speed-capacity-comparison.md` | [documentation/evaluation/model-speed-capacity-comparison.md](model_evaluation/model-speed-capacity-comparison.md) | `8a7683f00332ad7b316427d2839ba3c835277e61838157f97def34dfababcc4d` | `d834c31f276e33212031bb72de7e859d55291f419937821b5302f8800fd8e970` |
| `documentation/model-service-benchmark.md` | [documentation/evaluation/model-service-benchmark.md](model_evaluation/model-service-benchmark.md) | `0d8ec351dabb4a126d8a50440108668379999499104671595824e281feb9473f` | `be10d005802e53c198d604a1f5dbf9c752ad7c05759a6bc5b9487d140830d959` |
| `documentation/chat-model-benchmark.md` | [documentation/evaluation/chat-model-benchmark.md](model_evaluation/chat-model-benchmark.md) | `d5a2ab559cb7d2dbabcc1491608b158b7204ada99ba15f0be8d6361b09eabef6` | `3858a23718b984cba80546b3649d331a23e3b3428948844737aa8c2fb0710d3c` |
| `documentation/200条真人双标_训练评估实施方案.md` | [documentation/evaluation/200条真人双标_训练评估实施方案.md](model_evaluation/200条真人双标_训练评估实施方案.md) | `31a25e1d3b422cb6607542ef13f1f88d2a558b728e969c2d209a09dcf36c1734` | `6c8b44b05c7f15854702bdd52acf5d5fa8b99c70f59cc8d5d048019f31eee025` |
| `documentation/double-annotation-adjudication.md` | [documentation/adjudication/double-annotation-adjudication.md](double_annotation_adjudication/double-annotation-adjudication.md) | `c1c13fcf0dc2010af09609034941355e20ab1732df34fb8b1fbba526876db020` | `c1c13fcf0dc2010af09609034941355e20ab1732df34fb8b1fbba526876db020` |
| `documentation/double-annotation-workbench.md` | [documentation/adjudication/double-annotation-workbench.md](double_annotation_adjudication/double-annotation-workbench.md) | `c70f17dd7dd7f670f366aacd52a7a720467586274b8ebba80c503a68e3e5f14f` | `966d781c2de6b76d82c2932d13c38662719af3d0a20b40437583306475888498` |
| `documentation/double-annotation-release.md` | [documentation/delivery/double-annotation-release.md](project_delivery/double-annotation-release.md) | `7ae611f16eaef8471e8580882578b4cc875d4177cffd70c2e112a2c226916553` | `7ae611f16eaef8471e8580882578b4cc875d4177cffd70c2e112a2c226916553` |
| `documentation/国家疾控局来访汇报_讲述提纲_v0.1.md` | [documentation/delivery/国家疾控局来访汇报_讲述提纲_v0.1.md](project_delivery/国家疾控局来访汇报_讲述提纲_v0.1.md) | `11d32a59070548a53d570c0d1fd482fc22111b5a2b305002578d1fe52c977002` | `b0682c02c370d9694dd407e64c3c8ddde9ac4dfd61c50e073321c07405f8b606` |
| `documentation/国家疾控局来访汇报_讨论底稿.md` | [documentation/delivery/国家疾控局来访汇报_讨论底稿.md](project_delivery/国家疾控局来访汇报_讨论底稿.md) | `a98df4d84a3234326be18a4c2964abbb13e462c79fc1f079c0b87763b74687ae` | `5b828b8fd01bdf02a51b612171454c4e397e30237ac035ce111ee2ef9bb3f29b` |
| `documentation/国家疾控局来访汇报_待补事项与定稿条件.md` | [documentation/delivery/国家疾控局来访汇报_待补事项与定稿条件.md](project_delivery/国家疾控局来访汇报_待补事项与定稿条件.md) | `d494d13017069f9079907619989469d53ad5be43adef0fb55b65143d07366b7b` | `44e5fbe5bd28889abb7361e50071a1f16dc708c18ca96f62de581ddb243ba546` |
| `documentation/label-studio-single-selection.md` | [documentation/annotation/label-studio-single-selection.md](annotation/label-studio-single-selection.md) | `b36e1c9c8cea1fc76af1377e3b13ce51f37ed5ba4cbe649385baf022868e778b` | `223ee1d999709e4b8caa5a5bc1b56d3f5c52f40867b2f621780cbd4db6a55daa` |

SHA-256 不包含旧导航页；前后不同表示本地链接或后续目录说明发生变化，不代表冻结数据被重新生成。本表固定记录阶段 1 完成时版本；阶段 2 更新模板维护路径等正文后可与当前哈希不同，不回写阶段 1 快照。

## 保留项和后续范围

以下为阶段 1 结束时的保留决定；阶段 2 的实际完成项见后面的阶段 2 记录。

- 原始数据、模型权重、当前模型路径、历史轮次、指南历史、验证记录和 XML archive 不搬迁。
- `documentation/evaluation_examples/` 不修改、不重跑；其旧工具说明链接通过兼容导航继续可用。
- 顶层来源 PDF、识别用语提取稿与汇编暂留原位，待来源/维护稿分类后单独治理。
- `tmp/`、`data/`、`output/` 仅由主任务清点路径；本阶段未阅读病历内容、未移动或删除其中材料。
- 业务代码提取和模型服务拆分属于后续阶段，使用现有 CLI 及环境；本记录不宣称其已完成。

## 验证

- `python documentation/annotation/scripts/check-document-links.py`：通过，annotation 本地文件与静态 HTML 锚点存在。
- 本轮移动正文、旧导航页、README、AGENTS、CLAUDE、组织方案、文档总导航及迁移记录共 41 份文件的 271 个本地 Markdown 文件链接：全部存在；此项不检查互联网、所有 Markdown 章节锚点或全仓库历史链接。
- 对 `documentation/evaluation_examples/`、workflow 的 `runs/`、`guides/`、`validation/` 及 `label_studio/archive/` 中 160 份文件逐个比较迁移前后 SHA-256：全部相同。
- 在系统临时目录使用纯虚构 HTML、核查报告与空 JSON 模板实际调用 `scripts/package_double_annotation_review.py` 的 `package()`：读取 stage 正文，包内旧/新裁决方案链接及工作台发布说明链接正确，网页副本逐字节一致，两份 ZIP 生成成功。临时演示材料已清理，没有读取或打包本地真实病历。
- `git diff --check`：通过；Git 的 LF/CRLF 转换提示不影响该检查结果。

文档与打包检查不代替模型、服务、医学或 UI 验收。后续源码阶段验证分别记录，不将阶段 1 的测试数量作为最终数量。

## 阶段 2：无模型共享业务提取（已完成）

完成时快照：[source-migration.json](source-migration.json)。包含 14 项旧脚本实现→业务模块移动、1 项 HTML 源码模板移动，另记录两份新提取模块及三个消费者更新。来源哈希读取迁移前实际工作区，保留已有未提交改动；实现与兼容入口哈希记录阶段 2 完成时版本，后续阶段另记，不覆写此快照。

共享业务分为 `data_preparation`、`annotation`、`annotation_analysis`、`adjudication`、`training_data`、`evaluation`。14 个旧脚本保持 CLI 及导入兼容，通过模块别名返回同一个实现对象；私有符号与 patch 行为保持。`train_frozen.py`、`train_adjudicated.py` 及 smoke 工具直接引用新共享模块。正式逻辑只在业务包维护，旧脚本不保留第二份业务实现。

Schema 读取在 `emr_annotation/annotation/schema.py` 集中维护，保留导出分析 `load_export_schema` 与双标复核 `load_double_annotation_schema` 两种不同契约；`read_label_groups` 保留后代 Label 解析和原有校验。Choice 显示值/别名、单选合同和历史控件匹配保持。

HTML 模板迁至 `emr_annotation/adjudication/templates/double_annotation_review.html`，模板内容逐字节相同；渲染提取为 `workbench.py`，打包实现为 `package_review.py`。报告路径按实际输出目录计算，跨盘输出使用文件 URI；打包可将不同深度的新旧裁决方案链接转换成包内路径。

顺带修复已有报告显示边界：无可比病例时 kappa 返回的指标对象和空实体 F1 保持 None，报告显示“不可评价”，仍完整生成 audit、页面、核查报告及两份裁决模板。只调整显示，不更改评分、分母或原始审计 payload，也未重新生成历史结果。

实际验证：

- 阶段 2 完成时 `python -m unittest discover -s tests -v`：137 项通过，包含 9 项迁移专项检查。
- `python -m unittest discover -s tests -p test_shared_code_migration.py -v`：9 项通过，覆盖模块别名、新旧导入顺序、私有符号和 patch、无 site-packages/模型依赖导入、CLI 参数、虚构评分及训练转换等价、模板和打包、缺失病例选择时完整生成材料。
- 12 个旧 CLI 与 12 个新模块 CLI 的帮助参数一致；虚构实体评价输出字节一致，训练转换与冻结预检保持数据合同，双标 payload/HTML/模板一致。
- code-quality、anti-pattern 复核完成；复现的缺病例选择场景修复后，项目/总体病例指标仍为 null，输出 5 项材料及全部裁决项，评分语义未变。
- 本阶段没有执行真实模型训练、推理、服务联调、Label Studio UI 或医学验收。

阶段 2 文档同步：根规则、README、组织方案、总导航和工作台说明已指向当前共享模块及模板；文档命令保留旧 CLI。阶段 2 完成时阶段 3 正在进行、阶段 4 尚未开始；后续结果分别追加，不回写阶段快照。

## 阶段 3：模型源码分层（源码及离线验证已完成）

完成时快照：[model-migration.json](model-migration.json)，记录迁移前工作区源码、16 个原定义的 AST 哈希、25 份保护文件、四个实际实现及 package 初始化文件、两个旧兼容入口、五个后续消费者的完成时哈希。阶段 2 的 source-migration 快照保持原样；本阶段的训练 runner 更新在模型快照记录。

| 职责 | 当前实现 | 兼容关系 |
|---|---|---|
| 联合网络 | [modeling/network.py](../modernbert_ml_backend/modeling/network.py) | `ModernBERTForNERRE` 供训练和服务共同使用 |
| 推理解码 | [modeling/inference.py](../modernbert_ml_backend/modeling/inference.py) | `predict_text`、`NERREPredictor` 在 service 导入，旧 model 模块仍可访问 |
| Label Studio 服务 | [backend/service.py](../modernbert_ml_backend/label_studio_backend/service.py) | 旧 `model.py` 是此模块别名，`_wsgi.py` 的旧导入保持 |
| 训练实现 | [training/trainer.py](../modernbert_ml_backend/training/trainer.py) | 旧 `train_ner_re.py` 保留模块别名和 CLI，独立训练 runner 保留 |

trainer 直接引用 network，导入训练不加载服务或 Label Studio SDK；service 的 fit 在取得可用训练数据后延迟导入 trainer。内部继续使用顶层 `modeling`、`backend`、`training` 和同一个 `config` 模块，不是将服务迁成完整 `modernbert_ml_backend.*` 包启动。三个子包的入口为空，导入包本身不自动导入 ML、服务、config或联网。

`ModernBERTModel.setup.root_dir` 在新位置增加一层 dirname，仍得到迁移前的 `modernbert_ml_backend/` 根目录；仅此路径表达式和 fit 内部的 trainer 导入是记录允许的定义 AST 差异。模型/config/runtime/入口的既有工作区修改不以 Git HEAD 替换。本地 `label_studio_ml/`、config、requirements、基础权重及既有 best_model 产物受哈希保护，不移动、不恢复旧版、不合并环境。

训练 receipt 的代码哈希覆盖真实 network/inference/trainer/service 实现及兼容入口，不仅记录 wrapper。冻结训练的 AST 模拟测试读取实际 trainer；裁决训练 fake 保留旧模块并增加新网络模块 fake，不将模拟训练当成真实 ML 验收。

实际验证：

- 核心拆分全量 `python -m unittest discover -s tests -v`：146 项通过；模型迁移专项 9 项通过。补充 receipt 哈希断言后，模型/冻结/裁决相关 28 项通过；最终全量数量由阶段 5 收尾另记。
- 16 项类/函数定义的归一化 AST 保持；25 份 config、requirements、服务入口、本地 runtime、基础/既有训练产物文件的 SHA-256 保持。
- 新旧模型/训练模块别名、私有符号与 patch、新旧导入顺序、config 单例及服务根目录保持；原训练 CLI 参数及虚构调用、fit 延迟导入、训练不依赖服务/SDK、本地 runtime 来源检查通过。
- 阶段 1/2 索引覆盖的 160 份 workflow 历史、XML archive 和冻结示例文件逐个 SHA-256 保持；语法检查与 `git diff --check` 通过。
- 独立 verification、code-quality 与 anti-pattern 复核无阻塞；路径语义按迁移前实际工作区及 AST 核对，没有按 Git HEAD 改写用户已有实现。

实际模型环境尝试失败：`output/2026-10-10-code-migration/evaluation/model-separation-smoke-01/receipt.json`（本地 ignored 回执）。根 `.venv` 的 PyTorch C 扩展在导入时失败，回执为 failed；未执行网络 forward、文本解码、训练或服务。根环境元数据为 Torch 2.10.0 / Transformers 5.0.0，ModernBERT requirements 要求的独立环境未完成验证。没有改动依赖、权重或环境以掩盖失败，也没有把虚构 stub 通过写为真实模型、服务、UI 或医学验收。

阶段 3 文档同步：当前职责/目录树及规则已指向四个真实实现，旧 model/train 文件明确标为兼容入口；本地 runtime 和脚本命名空间约束保持。阶段 4 的实际复制、索引和保留范围见后续记录；阶段 5 的最终结果见后续记录。

## 阶段 4：明确旧材料副本归位与保留索引（已完成）

完整分类、来源原件、归位目录、兼容例外及恢复说明见[旧资料索引](legacy-materials.md)。本轮采用复制归位并保留原件，不是清空旧目录或改写历史。

| 分类 | 件数 | 新位置（相对仓库根目录） |
|---|---:|---|
| 指南阅读/打印核查截图及 PDF | 13 | `output/legacy-annotation-guide/delivery/run_20261010_stage4_144604/` |
| 来源 PDF 页面图/文字提取 | 54（52 JPG、2 TXT） | `output/legacy-source-documents/research/run_20261010_stage4_144604/` |
| 既有 WHO 采集成果 | 3 | `output/legacy-who-don-2026-06-01/research/run_20261010_stage4_144604/` |
| 外部来源 PDF | 2 | `data/legacy-source-documents/raw/run_20261010_stage4_144604/` |
| 外部人工筛选决定 | 1 | `data/emr_data_1_20261008/decisions/run_20261010_stage4_144604/` |

本地 manifest 位于 `output/legacy-project-migration/delivery/run_20261010_stage4_144604/manifest.json`，记录双方路径/大小/哈希、Git 跟踪状态、代码版本及工作区状态、保留索引和验证。清单和副本随 data/output 被 Git 忽略，未把约 10 MB 清单纳入源码版本管理；其他 clone 不保证有这些资料，须随材料另行备份。文档中的 ignored 目标及未跟踪来源使用代码路径，受跟踪的来源可保留链接。

实际验证与限制：

- 73 件副本全部在新的运行目录建立，来源/目标字节数与 SHA-256 一致；251 个保护来源前后大小/哈希相同。全部 86 件 tmp 原件及其 79 tracked / 7 untracked 状态保留，160 件冻结示例、历史和 XML archive 保持原内容。
- 已有 34,693 件文件（1,007,724,370 字节）的索引只读取路径、大小、用途和 Git 状态；未解析病例、模型权重，未执行旧 tmp 脚本、notebook或旧采集流程。本轮不批量复制原始 EMR或模型权重。
- 阶段 4 完成时独立复验 73 项副本和 251 个保护来源；当时旧资料页的 160 个本地文件/目录链接存在，`git diff --check` 通过。后续可移植性编辑将 ignored/未跟踪位置改为代码路径，此数量作为完成时快照保留，最终文档检查另记。
- 旧 tmp、冻结 manifests/轮次/历史、现有模型路径、根目录原始导出和两份混合来源/维护 Markdown 是有意保留项。15 件一次性脚本只建索引，未来需复用时单独核对后迁入业务模块；不声称所有旧目录已完成物理迁移。
- 若副本缺失，先核对 manifest 中原件字节数/哈希，再复制到新的 run_id 并校验，禁止覆盖旧结果；若原件缺失，只在原路径不存在且副本校验通过后另存恢复。双方缺失需从备份取回；历史冻结来源路径不重写。

阶段 1–4 实施范围已记录；阶段 5 的最终结果见后续记录。指定环境真实模型/服务、UI及医学验收仍未完成，不以复制和静态检查代替。

## 阶段 5：本轮范围整体验证与收尾（已完成）

本次源码/文档组织及明确材料副本归位的范围已完成；旧目录原位保留项和真实模型验收边界继续有效。

- 当前工作区 `python -S -m unittest discover -s tests -v`：144 项永久离线测试通过；annotation 本地文件/静态 HTML 锚点及 `git diff --check` 通过。
- 系统临时目录中的净源码沙箱只含 201 件源码、配置和文档，排除 data、output、bert-base-model、.venv、.git 和 tmp，清除 PYTHONPATH 后，以 `python -S` 完整运行同一套 144 项测试通过。证明离线测试不依赖本机 ignored 权重或旧结果。
- 阶段 3 的两项一次性 AST/25 本机文件哈希审计从永久测试移出：历史源码相等性与本机权重保留由完成时快照追溯，不要求后续合法实现变更永远匹配旧 AST，也不要求新 clone 携带模型权重。永久模型迁移测试保留 7 项运行/命名空间契约；阶段 3 原 146/9 项数量和来源哈希保持历史快照，不回写。
- [模型快照](model-migration.json)的 followup_validation 追加当前测试哈希、144 项工作区/净源码验证及回执；原 source/implementation/consumer 快照保留。独立复核代码哈希、73 对副本/251 项保护来源及 17 个旧导航页，通过；不将阶段 2 被后续修改的消费者哈希冒称为当前版本。
- 本轮当前维护 Markdown 文件链接存在性复核通过；ignored 运行回执/副本及未跟踪来源用代码路径表示，另说明其他 clone 可能缺失。维护汇报稿的本地批次汇总引用亦作此调整。历史冻结文档不改写。

实际模型环境仍是失败记录：根 `.venv` 导入 PyTorch C 扩展失败，没有实际 forward、文本解码、训练或服务验收。指定 requirements 环境、Label Studio UI 和医学复核需后续独立任务，不属于本轮离线迁移已通过的结论。没有提交、推送、删除原件、移动权重或合并环境。

## 后续调整：目录名称按用途统一（2026-10-10，已完成）

本轮重命名 7 个当前维护目录，共 28 个源码/技能/文档文件；完整映射见[目录命名规范](directory-naming.md)，逐文件路径和哈希见[目录调整记录](directory-renaming.json)。本页以前各阶段的路径标签、来源哈希和验证数保留历史语义，仅将导航链接指向当前正文。源码及模型 JSON 快照逐字节保持，不重写当时路径。

- `skill_src` 改为 `annotation_skills`；模型中的 `backend` 实现改为 `label_studio_backend`。`backend/` 仅留薄兼容层，`model`、`backend.service`、`label_studio_backend.service` 保持同一模块对象，config 单例、私有符号与 patch 行为保持。服务实现内容逐字节相同。
- 文档 `training_data`、`training`、`evaluation`、`adjudication`、`delivery` 分别改为 `training_data_preparation`、`model_training`、`model_evaluation`、`double_annotation_adjudication`、`project_delivery`。当前导航、相对链接、报告和打包工具同步；顶层旧文档导航页及已有 CLI 保留。训练回执同时记录新服务实现和旧兼容层哈希。
- `AGENTS.md` 增加目录命名规则，README、CLAUDE、组织方案及文档总导航同步。运行清单仍使用原 stage 标识，不跟随文档目录改名。
- 当前工作区 `python -S -m unittest discover -s tests -q`：144 项通过；另在不含 data/output/权重/环境/tmp/.git 的 206 文件净源码、配置和文档沙箱中，清除 PYTHONPATH 后，同一套 144 项通过。
- annotation 文件/HTML 静态锚点检查、当前维护 Markdown 文件链接检查与 `git diff --check` 通过。190 个历史、配置、runtime、权重及前轮 JSON 快照保护文件 SHA-256 保持。没有运行真实模型、训练或服务，也没有移动原始数据、冻结批次或模型路径。

本轮验证回执和改名前维护文件备份位于本地 `output/2026-10-10-directory-renaming/` 相应 delivery/evaluation 运行目录；它们受 Git 忽略。目录调整 JSON 中记录精确运行编号及路径，需要时随资料另行备份。

## 后续调整：data 顶层用途/批次/版本命名（2026-10-10，已完成）

用户指定的 data 六个顶层目录已实际重命名，完整原→新映射及版本规则见[数据目录索引](data-directory-naming.md)。名称采用明确用途、范围/批次、日期和 `v001`；版本号表示首次登记的存储快照，内部 UI v2、导出时间、候选 Schema、人工复核记录等原有含义保持。

- 六个新目录正常可见，六个旧名称保留为隐藏 Windows Junction，旧/新路径访问同一物理文件；不是额外副本。规范化路径会变成新名称，不保证路径字符串等同。后续数据清点跳过 Junction，避免重复统计；换工作区路径或恢复备份须核对本地兼容入口。
- 65 件原目录文件及 16 份根目录 CSV 的大小/SHA-256 保持；166 份仓库历史、冻结和前轮 JSON 快照保护文件保持。没有重写原数据、内部 manifest/receipt、HTML、ZIP、Schema、历史轮次或旧材料复制清单，也没有去重或合并七项目/八项目/试标导出。
- 当前复核与发布说明使用新 data 路径；旧 tmp 脚本不改写，通过兼容入口仍可访问相同材料。AGENTS、CLAUDE、README、目录命名规范、组织方案、文档导航及旧资料索引同步，data/README 仅为本地导航。
- 本次当前工作区 `python -S -m unittest discover -s tests -q` 144 项通过，annotation 文件/HTML 静态锚点及 `git diff --check` 通过，当前维护 Markdown 文件链接检查通过。没有执行真实数据转换、模型训练、服务、UI 或医学验收。

完整逐文件清单与兼容/哈希验证位于本地 `output/2026-10-10-data-directory-renaming/delivery/run_20261010_164840/manifest.json`，受 Git 忽略，需随数据备份。以前迁移记录及清单的路径/哈希仍表示当时版本，旧入口保证可访问，不将本次整理伪装成重跑历史任务。

## 2026-10-10 基础权重根目录命名

基础权重根由 `bert-base-model/` 改为 `pretrained_models/`；更新 config、裁决训练预检及虚构冒烟构建工具。现存 5 件 tiny 虚构模型文件逐件大小及 SHA-256 不变，旧根保留本地隐藏 Junction。训练 output、历史快照和冻结记录未改写。源码根 `modernbert_ml_backend/` 保留现名及脚本导入方式；真实模型/服务验收仍未完成。详见[改名记录](model-directory-renaming.json)。

本轮提交候选在不含数据、权重、output 和 tmp 的独立源码沙箱中通过 144 项离线测试及标注文档链接检查；配置路径方法单独验证基础权重新根和训练输出原路径，旧/新权重路径的 5 件文件物理身份及哈希通过。虚构 demo 的配置与输入采用 `.gitattributes` 固定 LF，并更新对应虚构选择凭据，避免 Git 换行转换导致来源哈希漂移；未改写真人冻结记录。
