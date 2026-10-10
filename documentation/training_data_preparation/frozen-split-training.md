# 使用冻结的患者分组清单训练

更新：2026-10-08。已实现[独立训练入口](../../modernbert_ml_backend/train_frozen.py)、[纯标准库数据核查与转换](../../emr_annotation/training_data/frozen.py)，并在训练模块中新增`train_from_splits`。2026-10-10 源码迁移后，[实际训练实现](../../modernbert_ml_backend/training/trainer.py)直接依赖网络，旧 `train_ner_re.py` 与 scripts 转换入口保留兼容；本轮完成虚构数据核查和训练函数模拟测试，没有运行真实训练或完成实际权重重载验证。

此入口解决“已按患者分组，但训练时又被重新随机切分”的问题。它读取明确的训练与验证集合，保留已审核无实体负例，不重新分组或自动过采样；测试集只参与归属与泄漏核查，不传给训练、标签权重计算或验证加载器。项目原`fit()`及`train_from_data()`仍沿用原路径，正式独立实验应选用新入口。

## 数据前提与核查

输入为[导出选择、医学裁决和患者分组流程](label-studio-evaluation-preparation.md)形成的整批`reference.json`及`train.json`、`validation.json`、`test.json`、`split_manifest.json`，同时提供冻结XML。

新入口逐项核对：

- 原参考与XML的SHA256匹配划分清单。
- 每个任务恰好出现一次，集合归属、原文校验值及完整参考字段与冻结源一致。
- 三份集合均非空，患者和精确全文重复没有跨集合。
- 实体偏移合法；训练/验证实体ID唯一；关系端点存在、类型属于XML。
- 训练和验证的字符级重叠实体被明确报错，避免单层BIO标签被后写入的实体静默覆盖。
- 同一有向实体对的重复或多关系类型被报错，避免单标签关系头覆盖答案。

关系方向`right`保持、`left`交换端点、`bi`展开为两个有向关系。实体ID和原文偏移保留；缺少ID且没有对应关系时可生成局部ID。属性和病例选择保留在原参考文件中，不作为当前NER+RE模型的训练目标。

这些检查不证明医学裁决已经充分，也不检查近重复、测试集是否此前用于调参、token边界、token重叠或截断丢失。字符无重叠不等于token一定无冲突；真实训练前需检查同版本tokenizer的表示能力和最大长度。遇到合法嵌套实体时，应讨论模型或监督策略，不能为了让检查通过而随意删改医学参考。

## 先运行数据核查

以下使用项目根目录；示例输入路径需真实数据到位后生成：

```powershell
python modernbert_ml_backend/train_frozen.py `
  --splits-dir output/evaluation/splits `
  --reference output/evaluation/reviewed/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/evaluation/training_preflight_001 `
  --check-only
```

核查模式不导入Torch/Transformers、不加载tokenizer或权重、不启动训练，只输出`training_receipt.json`。记录实际任务/患者/唯一文本、负例数、逐标签分布、输入与工具校验值、拟采用的参数和schema标识。它不产出模型或性能数字。

重复核查选择新目录；即使上一目录只有失败/核查记录也不覆盖。核查通过后，真实训练使用另一个新目录。

## 真实训练

在满足`modernbert_ml_backend/requirements.txt`的独立环境中运行。当前入口采用已配置的本地基础模型初始化和LoRA训练，使用全部XML实体及关系标签，并记录具体映射；它不是已定义的“训练前任务模型”对照。若对照要求从现有任务模型继续训练，需在基线确认后明确初始化协议。

```powershell
python modernbert_ml_backend/train_frozen.py `
  --splits-dir output/evaluation/splits `
  --reference output/evaluation/reviewed/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/evaluation/training_run_001 `
  --base-model chinese-modernbert-large-wwm `
  --epochs 20 --batch-size 4 --max-length 1024 --seed 42
```

模型名与参数只是现有默认配置的示例，应按实际可用基础模型、文本覆盖及验证集确定。记录随机种子不代表跨硬件完全确定。真实权重与运行环境尚未取得，本轮没有执行该训练命令。

训练复用原联合模型、数据集、LoRA配置、动态填充、损失和验证逻辑，增加显式数据入口和验证历史回调。新入口要求新输出目录，避免意外续训或覆盖已有模型；缺少LoRA依赖时明确失败，不悄然换训练方式。首轮验证即使指标为0也会保存候选产物，后续按原改进阈值及早停逻辑选择；原入口默认保存策略保持原样。

训练进度和结果：

| 文件 | 用途 |
|---|---|
| `training_receipt.json` | 数据、版本、参数、状态、实际选中验证轮次及产物校验值 |
| `training.log` | 已有训练过程日志 |
| `validation_history.jsonl` | 每轮训练/验证指标、综合选择分及是否选中保存 |
| `best_model/` | LoRA adapter、分类头、训练checkpoint及额外标签映射记录 |

失败保留回执和已有日志，不写成功状态。训练完成后会检查LoRA产物文件齐全并计算SHA256；文件齐全不证明权重可正确重载或模型已通过独立评价。综合选择分仍是token级NER macro F1与标注实体对RE F1之和，不能写成百分比准确率；回执区分训练器报告的最高分与实际保存轮次。

## 接入预测与汇报

本入口的示例输出目录为隔离实验目录，现有服务不会自动寻找它。服务按基础模型及标签schema定位权重，发布前必须核对实际加载路径、标签顺序、映射及日志，并完成真实重载测试。若选择服务原生的`output/{base_model}/{schema_id}`为训练输出位置，目录也必须全新；已有历史产物应保留，不能为通过入口检查删除它们。

完成独立模型验证后，用[服务预测与测速工具](../model_evaluation/model-service-benchmark.md)对同一`test.json`生成原始预测，再与[冻结正则和训练前模型](../model_evaluation/regex-baseline-comparison.md)逐标签比较。模型训练完成、模型重载成功、独立效果合格及速度测量分别留证，不能互相替代。

汇报中当前可以表述为“已新增使用冻结患者分组的独立训练入口，完成虚构数据与函数模拟验证”。真实200条训练、产物重载、严格实体结果、端到端关系结果和速度仍为待测；小型判别目标仍待确认。

## 已验证范围

[测试](../../tests/test_frozen_training_data.py)覆盖冻结源/清单匹配、参考被改动、患者泄漏、XML标签顺序、无实体负例保留、关系方向和非法端点/冲突、核查模式及历史结果保护。训练函数模拟直接执行实际函数体，验证只传入显式训练/验证数据、不自动扩样，并核对新增首轮零分保存策略与旧默认策略的区别。模拟不验证GPU、反向传播、LoRA适配、模型文件重载或真实医学效果。
