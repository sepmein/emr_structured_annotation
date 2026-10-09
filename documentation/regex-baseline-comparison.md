# 正则、训练前模型、训练后模型：逐标签对照

更新：2026-10-08。已实现正则实体抽取、逐条CPU测速及三方逐标签评分。当前正则规则为v0.1.0草案，尚未经过真人数据验证。已运行的结果使用20条完全虚构的短文本，只证明演示流程可运行，不能进入真人模型性能结果页。

## 交付文件

- [规则配置：49个实体标签](../annotation_agent_workflow/rules/regex_entity_baseline_v0.1.0.json)
- [正则预测与测速代码](../scripts/regex_entity_baseline.py)
- [三方逐标签评分代码](../scripts/evaluate_entity_predictions.py)及[评价口径](model-evaluation.md)
- [模型服务预测与测速工具](model-service-benchmark.md)，生成直接兼容评分器的真实模型预测；当前仅完成模拟验证
- [完整Label Studio格式虚构样例](../tests/fixtures/label_studio_demo_export.json)及[导出到训练数据处理工具](label-studio-to-training.md)
- [从完整导出转换后运行的逐标签结果](evaluation_examples/label_studio_export_demo/comparison.md)、[完整统计JSON](evaluation_examples/label_studio_export_demo/comparison.json)、[测速记录](evaluation_examples/label_studio_export_demo/regex_benchmark.json)
- [早期标准化参考样例](../tests/fixtures/regex_entity_demo.json)及[原演示快照](evaluation_examples/regex_v0.1.0_synthetic/comparison.md)保留供追溯
- [正则测试](../tests/test_regex_entity_baseline.py)及[评分测试](../tests/test_evaluate_entity_predictions.py)

## 正式比较的口径

| 内容 | 正则 | 训练前模型 | 训练后模型 |
|---|---|---|---|
| 输入 | 同一冻结测试任务、相同原文 | 相同 | 相同 |
| 参考 | 独立人工裁决，保留阴性记录 | 相同 | 相同 |
| 每个实体标签 | TP、FP、FN、参考数量、P/R/F1 | 相同 | 相同 |
| 辅助准确率 | 每条病历是否提及该标签 | 相同 | 相同 |
| 变化 | 比较基线 | 相对正则的变化 | 相对正则及训练前的变化 |
| 速度 | 相同计时边界的延迟与吞吐 | 相同 | 相同 |
| 可追溯信息 | 规则版本、校验值、配置 | 模型产物及配置 | 模型产物、训练清单及配置 |

主表衡量实体起止位置和标签同时正确。辅助准确率只判断标签提及出现与否，例如“无发热”仍包含“发热”提及，不能据此认为患者当前发热。否定、既往、计划、时间归属、关系和病例判别不在本版规则及评分范围内。判别模型仍需明确任务后单独评价。

展示所有49类，但无测试参考实体的类写“本批不可验证”，保留误报数。模型不支持某类时应注明支持范围；未训练的随机任务头不能充当已可用模型。正则提供一个可复核的任务基线，不改变“训练前模型”仍需明确的事实。

规则只允许依据训练/验证集调整。最终测试前冻结规则文件、推理代码、XML及校验值，与模型一起保留版本；正式测试后修改规则应使用新的独立测试批次，不用同一测试集反复调优来报告最终性能。

## 初版规则如何工作

规则来源为当前标签字典与XML，不加载外部模型，不读取输入中的人工实体或病例结论。词表别名有限；病原体主要采用标签名称字面匹配。明确写出的否定、既往、计划提及仍可抽取。温度升高不会自动生成发热实体，精神欠佳不会自动生成意识障碍。

规则保存完整原文字符偏移，支持用命名捕获只输出实体而不输出前缀上下文。同一实体控件内重叠候选先按优先级、再按长度取舍，不同控件可以重叠；同标签相同跨度去重。例如影像规则命中“CT提示右肺炎症”时只输出“右肺炎症”，并优先于内部“肺炎”诊断候选；病原名称中的“肺炎”不作为肺炎诊断。

有限规则有已知边界：流行病学事件、影像上下文、指标数值绑定只是初版近似；跨句、罕见别名、书写变体和复杂嵌套可能漏检或误判。当前上下文排除窗口为前后24字符。规则配置覆盖49标签不代表已证明49类识别能力，正式使用前需以训练/验证数据和医学复核完善。

## 真实数据运行

先按[导出、裁决和患者分组流程](label-studio-evaluation-preparation.md)得到`test.json`。在仓库根目录执行：

```powershell
python scripts/regex_entity_baseline.py `
  --tasks output/evaluation/splits/test.json `
  --rules annotation_agent_workflow/rules/regex_entity_baseline_v0.1.0.json `
  --label-config label_studio/pneumonia_config.xml `
  --output output/evaluation/regex.json `
  --benchmark output/evaluation/regex_benchmark.json `
  --repeats 5 --warmup 1 --dataset-kind human_test_set

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

上述真实数据与模型预测文件目前尚未提供。模型预测必须由实际模型产物产生，不能复制正则或参考结果充当模型输出；没有模型预测时省略对应参数，先形成正则结果。

正则测速逐条保存耗时、字符数、实体数和重复序号，输出P50/P95、平均延迟、成功吞吐及环境。一次预热不计时，文件读写、校验和规则编译不计入抽取延迟；编译耗时单列。任何抽取异常会终止运行，不生成成功批次。

当前测速是本地CPU抽取时间，不能直接和包含网络、排队的模型服务耗时计算提速倍数。正式速度表可分别列“本地抽取”和“完整服务响应”，仅在同一层、同一文本、同一硬件条件与并发设置下比较。正则扫描全文，模型截断时必须另报覆盖差异。虚构短文本的速度不能推及真实长病历。

[模型服务测速客户端](model-service-benchmark.md)已补齐单并发HTTP端到端测量，记录成功延迟、全部失败、吞吐和每次请求。该接口同时运行NER+RE，与规则实体抽取的功能范围也不同；尚未获得真实服务测量。

## 复现虚构演示

先将完整Label Studio格式样例转换为参考和训练文件，再将其中的标准化参考交给正则与评分器：

```powershell
python scripts/prepare_training_data.py `
  --export tests/fixtures/label_studio_demo_export.json `
  --label-config label_studio/pneumonia_config.xml `
  --selection tests/fixtures/label_studio_demo_selection.json `
  --ratios 0.6 0.2 0.2 --dataset-kind synthetic_demo `
  --output-dir output/label_studio_export_demo

python scripts/regex_entity_baseline.py `
  --tasks output/label_studio_export_demo/reference.json `
  --rules annotation_agent_workflow/rules/regex_entity_baseline_v0.1.0.json `
  --label-config label_studio/pneumonia_config.xml `
  --output output/regex_baseline_demo/regex_predictions.json `
  --benchmark output/regex_baseline_demo/regex_benchmark.json `
  --repeats 5 --warmup 1 --dataset-kind synthetic_demo

python scripts/evaluate_entity_predictions.py `
  --reference output/label_studio_export_demo/reference.json `
  --regex output/regex_baseline_demo/regex_predictions.json `
  --label-config label_studio/pneumonia_config.xml `
  --output output/regex_baseline_demo/comparison.json `
  --markdown output/regex_baseline_demo/comparison.md `
  --dataset-kind synthetic_demo
```

固定样例为人工指定的虚构文本及参考实体，不由本次正则预测生成参考答案；该批与规则均为本轮开发材料，样例并非盲测或独立医学验收。现在的入口为20条任务、40份标注的完整导出格式，经明确选择生成20条参考，与早期标准化样例的文本/实体相同。本处对全量20条进行接口演示，包含训练/验证部分，不是独立测试。正式评价改用冻结的`test.json`。

实际运行有20条任务、28个参考实体、24个有参考标签；正则TP=27、FP=0、FN=1，micro P=100%、R=96.43%、F1=98.18%。唯一漏检为“上月”时间表达，时间表达类P=100%、R=50%、F1=66.67%。其余25类没有参考实体，F1标“—”。这组高分只描述这些构造样例，不可当作真实任务准确率。

演示测速保存100次任务运行（20条×5轮），每次重跑耗时会变化；具体数值、运行时间与环境见配套测速JSON。当前尚无训练前或训练后模型实测，不提供三方性能胜负或提速结论。
