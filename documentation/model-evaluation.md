# 正则与训练前后模型逐标签评价

本工具用于国家疾控局来访汇报中的模型抽取结果页，使用Python标准库离线运行，不加载模型、不触发训练。它计算预测相对人工参考的指标；实际模型预测和医学裁决需另行完成。

入口：[scripts/evaluate_entity_predictions.py](../scripts/evaluate_entity_predictions.py)。

## 输入准备

准备同一独立测试集的四份文件：人工裁决测试参考`test.json`、正则预测`regex.json`、训练前预测`before.json`、训练后预测`after.json`。支持任意一至三份预测，缺失的运行不填结果。正则预测由[版本化规则工具](regex-baseline-comparison.md)生成。整批`reference.json`需先按[患者和精确全文重复分组](label-studio-evaluation-preparation.md)划分，不能整批同时训练和测试。

每份文件为任务数组，字段如下：

```json
[
  {
    "task_id": "fictional-001",
    "text": "虚构测试：发热",
    "entities": [
      {"start": 5, "end": 7, "label": "发热", "text": "发热"}
    ]
  }
]
```

上例完全虚构，用于说明格式，不是本项目真人标注或模型实测数据。

- `task_id`为非空字符串或整数。同一文件内不得重复，整数1和字符串`"1"`视作相同任务。
- `text`必须是未经重新拼接、清洗或换行替换的测试原文，所有运行与参考逐任务一致。
- `start`包含起点，`end`不包含终点；使用Python字符串字符位置。导出转换阶段须确认平台偏移与原文一致；包含非BMP字符等特殊文本时需核实偏移单位，不自动猜测转换。
- `label`必须来自本次锁定的XML实体标签。实体内`text`可省略，存在时必须与`text[start:end]`相等。
- 经过完整医学复核、确认没有目标实体的参考任务使用`"entities": []`。未完成标注的任务不能据此转换为负例。
- 所有正常预测必须提供`entities`，即使结果为空。失败预测可提供相同原文、`"entities": []`和`"status": "failed"`；正常状态为`"ok"`，可省略。
- 预测缺少整条任务时保留在评价分母并报告缺失；参考实体计入漏检。预测中多出的任务会报错，避免混入其他集合。
- 参考实体不得完全重复；重复预测通过一对一匹配计为额外误报，不自动去重以抬高分数。

这里的输入为标准化评价数组，不直接接受含两份`annotations`的原始Label Studio导出。可先使用[导出核查与参考准备工具](label-studio-evaluation-preparation.md)保留原始快照并明确选择医学裁决版；不得默认取第一份标注。参考准备工具同时保留关系、属性和病例选择，本评分工具只读取实体评价字段。整批参考还需划分后才能确定独立测试集。

实际ModernBERT服务预测可用[模型服务预测与测速客户端](model-service-benchmark.md)生成，保留首轮正式预测及失败状态，直接符合本评分格式。接口适配已用模拟响应和临时本机HTTP验证；真实模型服务仍需联调。

原始数据、标准化输入、预测与评价输出建议保存在受控的本地`output/`目录；该目录已被Git忽略。本工具不是去标识化工具。

## 运行

在仓库根目录执行：

```powershell
python scripts/evaluate_entity_predictions.py `
  --reference output/evaluation/splits/test.json `
  --regex output/evaluation/regex.json `
  --before output/evaluation/before.json `
  --after output/evaluation/after.json `
  --label-config label_studio/pneumonia_config.xml `
  --output output/evaluation/entity_comparison.json `
  --markdown output/evaluation/entity_comparison.md `
  --dataset-kind human_test_set
```

`--regex`、`--before`、`--after`至少提供一项。缺少训练产物时可只评正则，不用虚构预测凑齐三方。虚构数据演示使用`--dataset-kind synthetic_demo`，未核实来源时保留默认`unverified`；`human_test_set`仅为操作者声明，程序不会认证医学裁决。不需要阅读表时省略`--markdown`。输出路径不能覆盖输入，JSON和Markdown输出必须为不同路径。命令中的文件为约定位置，需先取得真实数据，当前不代表这些文件已经生成。

## 输出与指标

JSON包含输入文件名及SHA256、数据性质声明、配置分组、任务覆盖、逐标签原始计数、分组/总体指标，以及辅助的标签提及出现判断结果。`comparisons`保存训练前相对正则、训练后相对正则、训练后相对训练前的逐标签P/R/F1和辅助准确率差值；仅在双方输入均存在时生成，零分母不强行相减。保留原`f1_change`字段兼容训练前后对照。报告不复制病历原文，但保留缺失/失败任务标识以便内部排查，外部展示前应检查标识是否适宜。

Markdown包含所有实体标签，按XML原分组呈现：参考实体数、各运行TP/FP/FN、精确率/召回率/F1和各组F1变化百分点；同时展示辅助准确率、总体micro F1、macro F1及有参考标签数。当前配置为49个实体标签，其中`symptons_labels`为12个。

严格实体匹配要求原文起止位置与标签同时正确。边界或类别错误会产生预测误报和参考漏检。总体micro指标纳入全部标签误报；macro F1仅平均有参考实体的标签，并附有支持的类别数。

当某标签在测试集中没有参考实体时，召回率和F1记为JSON `null`、阅读表“—”；误报仍记录。没有预测时精确率为`null`；有参考但全漏检时召回率和F1为0。不能以无样本标签的显示值证明该类能力。

辅助准确率只回答“这条病历是否包含相应标签的提及”，不评价字符定位、否定状态、时间属性或患者当前是否有该症状。它保留所有参考任务为分母，缺失或失败响应均不获得正确判断的分数；对于失败且参考无该标签的任务，JSON以`failed_negative`单列。参考无阳性的标签即使辅助准确率很高，也不能据此声称已验证阳性识别能力。

## 结果进入汇报前

同时附模型版本、医学裁决记录、患者分组与重复排查清单，以及测试集类别分布。本工具不会自动证明参考经过裁决、训练与测试患者独立，或两个运行确为训练前后模型。

它不计算关系F1、属性效果、病例判别或运行速度；相应测量按[200条真人双标实验方案](200条真人双标_训练评估实施方案.md)另行准备。

## 已完成验证

新增离线测试覆盖严格跨度、错误标签、重复预测、无参考类别、缺失/失败任务与负例分母、原文不一致、重复ID、非法输入、输出覆盖保护、三方逐标签差值、仅正则评价，以及当前49标签配置下的完整命令输出。所有测试均使用虚构数据和临时目录。

运行项目离线测试与资料链接检查：

```powershell
python -m unittest discover -s tests -v
python documentation/annotation/scripts/check-document-links.py
```

离线测试通过仅说明这些评价逻辑得到验证，不代表真人标注质量、模型效果或速度已获得验证。
