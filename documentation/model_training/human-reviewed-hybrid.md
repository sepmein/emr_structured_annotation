# 人工裁决后mmBERT混合判断模型

本轮使用182例全部确认的人工裁决，保留原文、属性、关系及病例四分类。
当前流程入口为`modernbert_ml_backend/train_hybrid.py`、`predict_hybrid.py`，
生成与分组在`emr_annotation/training_data/human_reviewed.py`，证据特征在
`emr_annotation/evaluation/clinical_evidence.py`，报告在
`emr_annotation/evaluation/human_hybrid_report.py`。不会更换原Label Studio服务实现。

## 数据与参考边界

- 人工最终参考为`output/schema_training_trial_20261011_v001/adjudication/run_human_freeze_001/`，
  包含SQLite历史备份、最终导出、参考及XML/哈希。人工已确认，未获独立专家签署。
- 真实患者分组沿用`training_data/run_003`，128/27/27例；禁止用合成样本填充测试集。
- 生成器按Schema及裁决规范创建1000条、12种场景的样本，覆盖49类实体、5种关系，
  包括否定、既往、家属、条件/假设、计划、阴性、未知、儿童线索及明确结束时间。
  合成记录标为`synthetic_rule_adjudicated_not_human_gold`；规则逐点清单另存。
  这不是1000条新收集并由专家逐条确认的真实病例。
- 本轮最终数据是`training_data/run_hybrid_002`，训练1128、验证27、测试27。
  `run_hybrid_001`生成记录及先行训练保留为历史，不覆盖。
- 病例判断使用人工四分类。症候群辅助标签是人工实体属性派生的证据分组，
  没有独立症候群专家金标；病原标签是已知检验结果的检出分组，未知槽位不计损失。
  不能将检出解释为致病原因。真实关系无关系负例未全面标注，不报告完整关系P/F1。

## 架构与训练

先微调mmBERT最后4层及跨度实体/9属性/关系/病例任务头，保存并严格重载最佳抽取器。
再冻结抽取器，使用预测的全文CLS均值、文本病例概率与结构化证据训练门控融合器。
结构化特征包含实体断言状态、病原结果、关系类型计数和明确测量关系中的数值；
不会用最近数值猜测测量关系。测量控件没有症状断言字段，未标时间保留为未指定，
明确既往、计划、他人上下文的值不进入数值特征。

文本、证据、融合分支各有病例四分类、4个症候群证据与4个病原检出头，
分别以真实验证病例macro F1选择快照。这是共用抽取器的两阶段训练和分支比较，
不是三个独立编码器重训，也不是融合损失回传到mmBERT的端到端联合训练。
融合训练输入来自抽取器对训练记录的预测，当前没有交叉折外抽取。
人工记录损失权重5、合成1；辅助输出阈值固定0.5。测试只读task_id/text，忽略人工答案。

## 本轮命令与位置

使用独立GPU Python `modernbert_ml_backend/.venv/Scripts/python.exe`；根环境只负责stdlib数据准备和离线测试。
准备数据时目标目录必须是新run，不能覆盖冻结内容：

```powershell
& .venv/Scripts/python.exe -m emr_annotation.training_data.human_reviewed --help
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/train_hybrid.py --help
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/predict_hybrid.py --help
```

实际抽取器在`output/schema_training_trial_20261011_v001/training/run_hybrid_003`，
融合器在同一stage的`run_hybrid_fusion_001`。本轮训练/交付目录是到
`D:/codex_outputs/emr_structured_annotation/schema_training_trial_20261011_v001/`
相应stage/run的Windows Junction；备份或换工作区须保留真实D盘目标并重建入口。
模型配置记录抽取器绝对路径和权重哈希，不是可任意搬动的单文件模型。
历史训练轮次和输入保持原位置。

```powershell
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/predict_hybrid.py `
  --checkpoint output/schema_training_trial_20261011_v001/training/run_hybrid_fusion_001 `
  --tasks <只含task_id和text的JSON文件> --output <新的预测文件>
```

六部分报告、同批分支指标和推理核查在`delivery/run_hybrid_001`。
报告JSON保留来源哈希、指标分母与componentIds；网页使用Data报告运行时，
仅有聚合指标，不嵌入病历正文。独立新批次、折外特征、真实关系负例、
症候群/病因专家标注及在线服务接入仍需后续验证。
