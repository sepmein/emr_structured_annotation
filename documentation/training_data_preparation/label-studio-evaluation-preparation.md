# Label Studio导出核查与评价参考准备

本工具读取已经导出的完整JSON，统计双标情况、比较结果、检查部分字段，并根据明确的医学复核选择生成标准化参考。它不连接Label Studio服务，不自动裁决，不修改原始导出，也不执行去标识化或训练。

入口：[scripts/prepare_label_studio_evaluation.py](../../scripts/prepare_label_studio_evaluation.py)。仅使用Python标准库，既可脚本运行，也可被离线测试导入。

需要继续生成模型训练文件时，使用[导出到训练数据处理工具](label-studio-to-training.md)，可串联本核查流程、复核选择、患者划分及NER+RE转换；配有完整Label Studio格式的虚构样例及已运行结果。Choices的导出别名和中文显示值均按XML归一后比较，区域属性保留原始导出值。

## 第一步：保留原始导出并核查

取得包含每条任务`id`、`data`、`annotations`的完整JSON数组；每份标注保留`id`、`completed_by`、`was_cancelled`和`result`。不要用不包含两份原始标注的简化导出来代替。

配置须与本批标注时一致。当前配置读取`data.text`，原文与字符偏移不能经过重新拼接或替换。原始JSON、患者分组信息和后续参考均存放在受控数据目录，推荐使用被Git忽略的本地`output/`。

```powershell
python scripts/prepare_label_studio_evaluation.py `
  --export output/evaluation/label_studio_export.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/evaluation/inspection
```

路径为接收真实数据时的约定示例，当前不代表已有对应导出。

输出：

- `audit.json`：任务、原始标注、可转换标注、取消/无效标注的数量；每条任务候选标注ID；缺少患者分组字段的数量；可比双标的逐层精确一致情况及输入校验值。报告不复制病历原文。
- `selection_template.json`：每条任务默认`pending`，无默认标注选择、无复核确认。不会生成`reference.json`。

“可转换”表示通过本工具的有限结构检查，不表示医学正确或通过全部标注规范。当前检查实体控件和标签、原文跨度、属性指向、关系端点和类型，以及配置中必填的病例选择；不会完整判断条件显示与必填属性，也不能证明页面兼容性。

两份标注只有在均可转换、标注者ID已知且不同、任务恰好有两份可转换标注时才进入双标比较。不同账号仍不能证明两人未相互讨论或未受预标注影响，需另行核实流程。

一致比较按实体原文位置、标签及所挂属性/关系进行，不要求两人生成相同的区域ID。这里的精确集合一致率是描述性统计，不能当成模型F1或Cohen's kappa。病例、实体、属性及关系分别列出。

同一任务内重复的标注ID会使相关副本全部失去可选资格，避免指向含糊结果。平台`predictions`始终不作为人工参考来源。

## 第二步：医学复核与裁决

保留两份原始标注，由医学复核人员处理分歧并抽查一致任务。如果裁决需要组合或修改两人的结果，应在受控流程中形成单独的完整裁决标注，再导出包含该版本的JSON；本工具不会自动合并两套答案。

修改导出后须重新核查并使用新模板。模板绑定原始导出和配置的SHA256，旧模板不能直接用于新数据。涉及第三份裁决标注的任务可明确选择裁决版本，但不再纳入“恰好两份”的原始双标比较；初次核查报告单独保存。

对每个任务填写选择：

```json
{
  "task_id": "fictional-001",
  "action": "include",
  "annotation_id": "fictional-reviewed-annotation",
  "reviewed": true,
  "reviewer": "reviewer-01",
  "reason": "已复核，采用裁决版"
}
```

上例只说明格式，ID均为虚构。保留模板顶层的两个校验值和`selections`数组；仅在实际复核后填写确认。复核人员可使用内部代号。

需要排除的任务使用`"action": "exclude"`，同时填写`reviewed`、`reviewer`和非空`reason`。所有导出任务均须明确纳入或排除，不能在生成参考时默默丢弃未处理任务。仍为`pending`、指向取消/无效标注、缺少复核确认或校验值不符时不会生成参考。

无实体但经过医学复核、且具备当前配置要求的病例结论的标注可以保留为负例。完全空的`result`不自动转换为负例。

## 第三步：生成标准化参考

将完成复核的模板保存为例如`reviewed_selection.json`，使用另一个输出目录：

```powershell
python scripts/prepare_label_studio_evaluation.py `
  --export output/evaluation/label_studio_export.json `
  --label-config label_studio/pneumonia_config.xml `
  --selection output/evaluation/reviewed_selection.json `
  --output-dir output/evaluation/reviewed
```

输出：

- `audit.json`：本次原始导出的核查结果。
- `reference.json`：明确选择的参考任务，保留`task_id`、精确原文、患者分组字段、实体、属性、关系、病例选择和来源标注ID。实体字段可直接用于[逐标签评价工具](../model_evaluation/model-evaluation.md)。
- `selection_receipt.json`：绑定输入校验值，记录纳入、排除、选择的标注、复核人员及原因。

工具拒绝覆盖上述已有文件，不覆盖输入或上轮结果。不同批次及重复运行使用不同的输出位置。

`reference.json`是整批已选择参考，不应直接将全部内容作为独立测试集。下一步按患者分组并检查重复文本，生成训练、验证和最终测试清单，再训练与预测。

参考保留原文与患者分组字段，不是去标识化工具的输出认证。原始导出是否满足隐私要求须在项目既定流程中验收。

## 第四步：冻结患者与精确重复文本分组

已新增[独立数据划分工具](../../scripts/split_evaluation_reference.py)。它将同一患者的记录归组，也将不同患者间原文完全相同的记录连接到同一组；关联会传递，整个连通组必须留在同一个集合。

例如A患者有记录甲和乙，B患者的记录丙与乙全文相同，则甲、乙、丙均不能跨训练/验证/测试集合。空原文也按相同文本归组，不因内容少而当作多条独立样本。

所有任务必须带稳定、非空的`patient_id`；缺失时停止划分，不自动使用任务ID替代患者。只检查精确全文重复，近重复和相似模板仍须另行检查。

```powershell
python scripts/split_evaluation_reference.py `
  --reference output/evaluation/reviewed/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --ratios 0.6 0.2 0.2 `
  --seed 42 `
  --output-dir output/evaluation/splits
```

60%/20%/20%仅为待讨论的起始示例，命令要求显式指定三个正比例且总和为1。实际整组分配可能无法恰好达到比例；工具保留全部任务，三个集合非空，并报告实际条数和比例。如果不足三个独立组，停止划分。

输出`train.json`、`validation.json`、`test.json`及`split_manifest.json`。各数据文件保留完整参考字段，病例级选择仍作为原始类别保留，未预设小型判别模型的合并规则。清单记录输入与配置校验值、随机种子、每条任务和组的归属、患者/任务/唯一文本数量、无实体负例数及逐标签覆盖。

工具在写文件前核对患者与全文重复没有跨集合。当前分配方法使用固定种子和任务数量平衡，不保证所有标签或病例类别都覆盖三份集合；缺少标签样本会明确列出。整批未出现的标签单列，不能因为划分完成就声称可以评价49类。

输出位置必须没有同名结果，避免覆盖已冻结划分。确定最终测试集后，后续不得根据模型测试得分反复调换样本。输出组和文本校验值有助于内部追溯，不代表完整去标识化或跨来源泛化得到验证。

### 与训练及评价衔接

训练仅使用`train.json`，调参与模型选择使用`validation.json`；最终训练前后对比使用相同的`test.json`生成预测并交给逐标签评分工具。

现有ModernBERT `fit()`仍会拉取项目数据并自行随机切分。已新增[冻结清单独立训练入口](frozen-split-training.md)，核查源参考及清单、保留无实体负例，显式使用训练与验证文件；当前完成虚构数据及函数模拟验证，真实训练仍待开展。正式实验选择新入口后留存训练回执，不能仅生成清单后继续用旧入口训练，再宣称已按患者分组执行。

## 验证范围

离线测试使用虚构任务覆盖不同区域ID的一致比较、属性分歧、相同或未知标注者、取消/空白/无效结果、重复标注ID、错误控件、悬空属性与关系、病例选择缺失、复核选择与排除记录、过期校验值、不可覆盖历史输出，以及生成参考到实体评分的衔接。划分测试另覆盖跨患者相同文本的传递关联、患者和原文无跨集合交集、输入顺序无关的复现、负例及元数据保留、缺少患者信息与独立组不足、标签覆盖缺口和已冻结结果保护。

尚未取得本批真实真人数据，不能声称已完成实际导出核对、医学裁决、真人一致性测量、训练或模型性能评价。
