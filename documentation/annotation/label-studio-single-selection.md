# 实体单选配置与旧项目兼容

更新：2026-10-09。

[默认配置](../../label_studio/pneumonia_config.xml)保留9个旧实体控件名，全部显式设置 `choice="single"`，用于继续标注已有项目。每组内只选一个标签；不同组的标签可能同时处于选中状态，框选前请取消其他组的选中。时间表达按钮常驻标签区，便于切换后框选时间。

保留的控件名为 `symptons_labels`、`diagnosis_labels`、`epidemics_labels`、`time_labels`、`measure_entities`、`measure_labels`、`pathogen`、`bacteria`、`other_pathogen`。标签值、文本控件 `chief_complaint_text` 和 `$text` 绑定均保留。

[跨分组单选版](../../label_studio/pneumonia_config.global-single.xml)将全部49个标签放在一个 `Labels name="symptons_labels" choice="single"` 控件内，适用于新项目或明确迁移后的副本。其分组仅负责显示；从任一分组点击另一个标签，会取消前一个标签的选中。两版包含相同的标签、属性和关系，但实体结果的 `from_name` 合同不同。

一份病历仍可包含多个独立实体，例如分别标注“发热”和“新冠病毒”；切换当前标签不会删除其他文本区域已经完成的实体。选中已有实体时点击另一标签可能更改该实体的类型，请先取消实体选中再创建下一个实体。

病例四分类和9个实体属性字段仍按各自字段单选。不同属性字段可以同时有值，例如同一病原的检验结果、标本类型和检测方式；它们不是互相排斥的字段。实体ID、属性别名、关系和文本偏移合同保持不变。

依据：[官方 Labels 文档](https://labelstud.io/tags/labels)说明 `choice` 支持 `single` 与 `multiple`，默认 `single`；[官方单选实现](https://github.com/HumanSignal/label-studio/blob/develop/web/libs/editor/src/tags/control/Label.jsx)仅取消所属控件内的标签，因此原来的9个实体控件需要合并才能跨分组互斥。

## 已有项目与历史导出

已有项目保存跨分组单选版时，可能出现 `Created annotations are incompatible with provided labeling schema`，并列出旧控件名。这是实体控件合并后与已保存结果的 `from_name` 不匹配；仅调整 CSS 或 `choice` 无法修复。继续使用旧标注时，在项目 Settings → Labeling Interface → Code 中粘贴默认配置并保存，无需删除标注或改写导出。

修改前的配置完整保存在[旧配置快照](../../label_studio/archive/pneumonia_config.before-global-single.xml)。默认配置恢复了相同的实体控件与标签映射。历史导出和冻结清单仍应使用当时绑定的配置及哈希，通过现有转换命令的 `--label-config` 参数校验；不要覆盖历史试标、裁决和冻结结果。

如需跨分组单选，先导出并保留旧项目；在新项目中使用跨分组单选版，导入迁移副本，将已核对的实体结果的 `from_name` 映射到统一控件，同时保留ID、跨度、原文、属性和关系。多标签实体需人工复核，不能简单取第一个标签；旧选择清单和冻结清单的哈希不能直接用于新副本，需重新核查。项目迁移未在本次自动执行。

虚构测试导出使用跨分组单选版，演示命令须明确指定 `--label-config label_studio/pneumonia_config.global-single.xml`，演示选择清单绑定该文件哈希。这不是历史真人标注的迁移或医学复核。

## 验证范围

本地配置解析与离线数据测试可检查标签、属性、关系及转换合同，包含报错中五种旧控件名的11条虚构结果，不能证明实际项目保存和界面兼容性。正式更新项目时应在安装版本上检查配置保存、标签切换、属性面板显示、已有实体改标签、新实体创建，以及导出后回读。
