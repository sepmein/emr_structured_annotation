# Label Studio完整导出到模型训练数据

更新：2026-10-08。处理工具和虚构演示已实际运行；未使用真人数据训练模型。

2026-10-09：默认配置恢复9个旧实体控件名，各组内单选，兼容已有项目。虚构样例及其演示选择清单使用单独的跨分组单选版 `pneumonia_config.global-single.xml`；历史导出和冻结清单需使用原来绑定的配置。参见[单选配置与历史兼容说明](../annotation/label-studio-single-selection.md)。

入口：[数据处理工具](../../scripts/prepare_training_data.py)。仅使用Python标准库，可在仓库根目录运行。它串联导出核查、明确选择复核标注、格式转换、患者分组划分及训练前数据检查。

## 虚构样例采用完整导出结构

[20条虚构任务的Label Studio格式JSON](../../tests/fixtures/label_studio_demo_export.json)包含`id`、`data.text`、`annotations`、`predictions`等字段。每条任务有两份不同账号的标注，共40份；实体使用跨分组单选版的`from_name`、`to_name`、`id`、`value.start/end/text/labels`，同时包含区域属性、关系和病例选择。区域属性使用`known_absent`、`current`等导出别名。

结构参照[官方JSON任务格式](https://labelstud.io/guide/task_format)、[导出说明](https://labelstud.io/guide/export.html)及[Choices说明](https://labelstud.io/tags/choices.html)，并以[跨分组单选版XML](../../label_studio/pneumonia_config.global-single.xml)校验。这是模拟完整导出的测试文件，尚未通过真实项目界面的导出回读验收；取得真实导出后仍需核对平台版本和实际字段。

样例有意包含：第一条的首份标注遗漏发热，第二份包含发热；第14条有错误预标注，但人工标注无实体。这两种情况用于验证不会默认取第一份或采用`predictions`。两条无实体记录保留为训练负例。所有文本、患者、标注账号及复核信息均为虚构。

所有病例选择均为`待专业复核`占位，只用于检验字段传递，不是医学结论，也不能用于训练或评价病例判别模型。[样例选择清单](../../tests/fixtures/label_studio_demo_selection.json)明确选择每条第二份标注；其中`reviewed`仅模拟流程，不代表真实医学裁决。

## 真实数据处理：先核查，再转换

取得Label Studio完整JSON导出后，先运行以下命令。路径为约定示例，当前不代表已有真人导出。

```powershell
python scripts/prepare_training_data.py `
  --export output/evaluation/label_studio_export.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/evaluation/training_audit
```

此时只生成`audit.json`、待填写的`selection_template.json`及`conversion_report.json`，不会生成训练数据。模板默认每条任务`pending`；由医学复核人员根据原始标注填写`include`及`annotation_id`，或者填写`exclude`和原因，并记录`reviewed: true`及复核者。所有任务都须明确处理。具体字段与裁决方法见[导出核查及复核说明](label-studio-evaluation-preparation.md)。400份双标记录不能直接变成400条独立训练样本。

保留模板的导出/XML校验值及`offset_unit_in`，将完成复核的文件另存为`reviewed_selection.json`，再执行：

```powershell
python scripts/prepare_training_data.py `
  --export output/evaluation/label_studio_export.json `
  --label-config label_studio/pneumonia_config.xml `
  --selection output/evaluation/reviewed_selection.json `
  --ratios 0.6 0.2 0.2 --seed 42 `
  --dataset-kind human_reviewed `
  --output-dir output/evaluation/training_prepared
```

分组以稳定的`patient_id`及完全相同原文为约束，两者形成的关联组不能跨训练/验证/测试。比例为目标比例，实际数量受组大小影响；三个集合均须非空。没有患者字段时先补充分组依据，不能退回逐条随机划分。近重复文本仍需另外检查。仅做格式转换时可省略`--ratios`，此时不生成划分。

每次使用新输出目录。工具先校验并在临时目录生成完整文件，通过训练数据预检后才发布最终目录，拒绝覆盖已有结果。它不连接平台、不去标识化、不自动裁决，也不启动训练。

## 输出与训练接口

| 文件 | 内容与用途 |
|---|---|
| `audit.json` | 可转换/取消/无效标注、双标比较及输入校验值 |
| `reference.json` | 每任务一份明确选择的参考；保留原文、患者、实体、属性、关系、病例选择及来源标注ID |
| `selection_receipt.json` | 纳入、排除及复核记录 |
| `training_data.json` | 全部纳入记录的NER+RE格式：`task_id/text/entities/relations`；不是独立测试集 |
| `splits/train.json`、`validation.json`、`test.json` | 保留完整参考字段的冻结划分，供训练入口及独立评价读取 |
| `splits/split_manifest.json` | 患者/原文分组、任务归属、标签数量与参考/XML校验值 |
| `model_data/train.json`、`validation.json`、`test.json` | 各集合转换后的NER+RE格式，可供现有`NERREDataset`使用 |
| `frozen_training_preflight.json` | 划分、文件绑定及训练格式检查结果；明确未训练模型 |
| `conversion_report.json` | 原始任务、纳入数量、负例/实体/关系数量、划分、偏移单位及来源校验值 |

属性中的别名归一为中文显示值，同时以`export_values`保留原始值。病例选择及属性保留在完整参考中；现有NER+RE训练数据只消费实体与关系，不能因此宣称已实现这些属性和病例选择的模型训练。

转换检查实体跨度与原文一致，关系端点存在，并按方向转换关系。当前单层BIO训练不能表示的实体重叠、同一有向实体对的重复/多种关系会拒绝转换，需复核或另行制定建模方案，不能静默删除。取消、无效标注和`predictions`不会作为参考；完全空白标注不自动当负例。

默认字符单位为Python Unicode字符位置。若实际导出对非BMP字符使用UTF-16单位，须从核查阶段开始显式设置`--offset-unit utf16`，转换阶段保持同一参数与模板单位。工具不会猜测；UTF-16代理对内部的非法边界会使标注无效。原文不做清洗、替换或重拼接，原始导出不修改。本批虚构中文文本不含非BMP字符，两种单位位置相同。

推荐使用[固定划分训练入口](frozen-split-training.md)，先做不加载模型的数据检查：

```powershell
python modernbert_ml_backend/train_frozen.py `
  --reference output/evaluation/training_prepared/reference.json `
  --splits-dir output/evaluation/training_prepared/splits `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/evaluation/training_check --check-only
```

实际训练使用ModernBERT单独的依赖环境及本地基础权重，去掉`--check-only`并选择新的实验目录。该入口只将训练/验证划分传入训练，不重新随机拆分。不要把全量`training_data.json`交给旧版随机拆分流程后，仍用这里的测试集声称独立评价。token对齐、截断、模型重载及性能需实际运行后检查。

## 已运行的演示与复现

```powershell
python scripts/prepare_training_data.py `
  --export tests/fixtures/label_studio_demo_export.json `
  --label-config label_studio/pneumonia_config.global-single.xml `
  --selection tests/fixtures/label_studio_demo_selection.json `
  --ratios 0.6 0.2 0.2 --seed 42 `
  --dataset-kind synthetic_demo `
  --output-dir output/label_studio_export_demo
```

实际结果：40份标注均通过有限结构检查，选择后20条记录、2条无实体负例、28个实体、5条关系，分为12条训练、4条验证、4条测试。可查看[转换报告](../evaluation_examples/label_studio_export_demo/conversion_report.json)、[完整参考](../evaluation_examples/label_studio_export_demo/reference.json)、[全量训练格式](../evaluation_examples/label_studio_export_demo/training_data.json)、[划分清单](../evaluation_examples/label_studio_export_demo/splits/split_manifest.json)及[训练数据预检](../evaluation_examples/label_studio_export_demo/frozen_training_preflight.json)。配套[离线测试](../../tests/test_prepare_training_data.py)覆盖别名、负例、错误预标注、选择与文件绑定、非BMP偏移、患者划分及输出保护。

还用转换后的全量20条参考实际运行了[正则逐标签流程演示](../evaluation_examples/label_studio_export_demo/comparison.md)。这一演示包含训练/验证部分，仅证明数据接口衔接；不是独立测试结果。真实逐标签评价必须使用冻结的`test.json`，同时保存正则、训练前及训练后模型的原始预测。
