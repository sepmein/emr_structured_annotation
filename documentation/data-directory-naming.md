# data 目录用途、批次与版本命名

整理日期：2026-10-10。本次只重命名六个已有顶层目录，并保留旧路径兼容访问；没有修改数据内容、删除重复导出、改写历史清单或迁移内部冻结材料。

## 当前目录

所有路径以 `data/` 为根。下列目录是本地输入/历史材料，受 Git 忽略；普通 clone 不保证存在，因此使用代码路径表示。

| 用途 | 原目录 | 当前目录 | 已有材料 |
|---|---|---|---|
| 8 项目双标导出快照 | `2026-10-10-double-annotation-8projects` | `double_annotation_exports_8projects_20261010_v001` | 8 份原始导出、1 份候选 XML；候选配置不是已经确认的各项目配置 |
| 7 项目导出及配置候选 | `batch_20261010` | `annotation_schema_candidates_7projects_20261010_v001` | 7 份导出、7 份项目候选 XML |
| 试标导出快照 | `batch_test_flight_20261010` | `trial_annotation_exports_20261010_v001` | 12 份导出；同项目不同时间的文件全部保留 |
| 首批双标复核历史 | `anotated_results` | `double_annotation_review_batch_001_20261009_v001` | `batch_1/` 内的原始导出、人工复核、工作台与 v2 发布包，共 27 件 |
| EMR 批次筛选决定 | `emr_data_1_20261008` | `emr_screening_decisions_20261008_v001` | `decisions/` 下的 1 份外部筛选决定 |
| 临床规则/监测方案参考来源 | `legacy-source-documents` | `clinical_reference_sources_20261010_v001` | `raw/` 下的 2 份来源 PDF；日期表示材料归档批次，不是文献发布日期 |

前两套导出材料分别保留，不能因为部分文件可能相同就合并成一套“最终数据”。目录用途来自已有目录、文件类型和批次信息，不认证标注正确性、项目配置一致性或医学复核状态。

## 新批次命名规则

采用 `<purpose>[_<scope>][_<batch_id>]_<YYYYMMDD>_vNNN`：

- `purpose` 明确用途，例如 `double_annotation_exports`、`trial_annotation_exports`、`emr_screening_decisions`、`clinical_reference_sources`。
- `scope` 只在确有帮助时添加，例如 `8projects`、`7projects`；不要使用患者姓名、证件号等信息。
- 多批次可用 `batch_001`、`batch_002` 区分；不要只写 `batch`、`test_flight` 或含拼写错误的 `anotated_results`。
- 日期取有依据的批次、导出或归档日期；资料发布日期不明时，不把归档日期写成发布日期。
- `v001` 是本次首次登记的存储快照版本。不是模型版本、人工标注质量版本、Schema 版本、指南版本或工作台 UI 版本。现有 `release_v2_*`、`audit.v2.json` 等内部名称保持原义。
- 新增或修订输入时建立 `v002`、`v003` 等新目录，记录来源/哈希/变更说明，不覆盖已冻结版本；同一输入版本的多次处理只增加 `output/<batch_id>/<stage>/<run_id>/` 的运行编号。
- 与 output 对接时引用精确 data 批次/版本及哈希；已有 output 批次不因本次目录改名而重命名，也不改写旧 receipt/manifest。

新批次内部沿用 `raw/`、`label_studio_exports/`、`schema/`、`decisions/` 等内容位置。当前 `schema_candidates/` 存放候选配置，须核对后才能作为项目绑定的 Schema。第一批复核目录中历史工作台/发布包是原有混合输入产物的兼容例外；未来新工作台、分析、训练和发布产物应写入 output。

## 旧路径兼容与遍历

六个旧目录名是本地 Windows 目录联接（Junction），指向上述新目录；旧入口设为隐藏，新目录正常可见。它们与新目录访问同一份物理文件，没有复制出第二套数据。历史清单、旧脚本和已发布材料引用旧名称时仍能读取原文件。

目录联接保证文件可访问，`Path.resolve()` 等取得的规范化路径会变成新名称；不能据此声称路径字符串完全没变。冻结校验继续以原文件内容/哈希为准，本次没有执行真实数据转换、模型训练或服务验证。

清点 data 时只遍历新目录，跳过 Junction/重解析点，避免把同一批材料统计两次。旧入口不作为新数据写入位置，不把它们当普通独立目录递归清理。data、兼容联接和本地完整清单不随普通 Git clone 提供；换工作区路径或恢复备份后，须按本页映射重新核对或建立指向当前 data 内对应目录的兼容入口。

根目录原有 16 份 CSV 保持原文件名和位置，保留现有数据合并默认输入。`data/README.md` 仅为本地导航，维护规则以本页与 [AGENTS.md](../AGENTS.md) 为准。

## 验证及记录

六个目录内 65 件文件，改名前后 SHA-256 和字节数一致；每件旧路径与新路径访问同一物理文件。16 份根目录 CSV 及 166 份仓库历史/冻结/前轮快照保护文件保持。

完整逐文件清单保存在本地 `output/2026-10-10-data-directory-renaming/delivery/run_20261010_164840/manifest.json`；含路径映射、用途、日期/版本含义、兼容入口、哈希和实际验证。该清单受 Git 忽略，需随数据备份，不将真实材料清单复制到测试夹具或源码中。后续收尾记录见[项目迁移记录](migration-log.md)。
