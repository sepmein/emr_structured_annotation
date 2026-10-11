# 完整Schema的中文跨度多任务试验

本入口用于明确标为未裁决的双人原始导出试验，不能代替最终医学金标训练。复用共享Schema解析、导出整合和患者/全文连通分组；原始文件不改写，两个标注者视图在同一集合。

## 实现与能力

- [数据转换](../../emr_annotation/training_data/schema_multitask.py)：无效选区所在视图屏蔽实体损失；仅保留合法的已知属性，缺失值不补成阴性。属性遵守XML作用域、别名和选区偏移。
- [网络](../../modernbert_ml_backend/modeling/schema_multitask.py)：本地中文BERT共享编码器、有限宽度多标签跨度头、9个作用域属性头、全文窗口病例四分类及实体对语境关系头。支持重叠/同跨度多标签，但不认证这些原始标签的医学正确性。
- [训练](../../modernbert_ml_backend/training/schema_experiment.py)：微调编码器后4层，验证集选模、保存和严格重载；实体阈值预先固定。属性提供人工实体条件及预测实体条件下已知槽位的正确比例；病例提供四类混淆与macro F1。
- [报告生成](../../emr_annotation/evaluation/schema_training_report.py)：六部分汇报、固定正则对照、49类逐标签与9属性附表、可核查指标快照。报告React内容由交付目录中的独立应用承载。

关系监督只使用原文件明确保存类型的正例。多标签区域只有在按Schema合法拓扑能唯一解析实体端点时才登记投影，保留原关系引用；没有类型或存在多个候选时屏蔽，不猜测类型。未标实体对不视为负例。关系分支按训练集合实际有支持的类型启用；预测是合法端点的类型候选，尚无边存在概率校准。完整P/F1需要完整正负参考，缺少验证/测试支持时保持不可评价，不能宣称医学效果验收。

首轮`run_001`采用唯一实体区域合同，关系分支停用；`run_002`保留原始明确类型的唯一Schema端点投影，实际训练关系头。两轮患者、原文和实体划分保持一致，旧结果不改写。当前保留关系正例恰好全部属于训练集合；关系泛化效果及无正例类型不能从训练参数更新推断。

实体最长32 token、窗口384 token且重叠96 token。窗口覆盖全文，但跨token边界或过长实体可能不可表示；这些实体保留评分分母。属性未标槽位不评分，不将已知槽位正确比例称作完整属性F1。病例分支使用窗口CLS均值，尚未融合结构化证据。

## 基础权重与命令

使用[官方中文BERT](https://huggingface.co/google-bert/bert-base-chinese)，精确下载revision及文件哈希保存在本地基础权重目录和实验回执。训练仍使用[独立GPU环境](runtime-environment.md)，不修改原ModernBERT服务及权重路径。

后续基座支持官方mmBERT，层冻结适配、紧凑checkpoint保存及同口径对比见[mmBERT与中文BERT](mmbert-base-comparison.md)。沿用最后4层微调；官方分词器、编码器规模与窗口监督覆盖作为明确差异记录，不把更换基座称为纯架构消融。

示例采用本轮登记的同名data/output批次，原始来源及字节一致快照见该data批次的snapshot_manifest.json。新运行须更换run_id，不能覆盖旧结果。

```powershell
& .\modernbert_ml_backend\.venv\Scripts\python.exe modernbert_ml_backend/train_schema.py `
  --input-dir data/schema_training_trial_20261011_v001/label_studio_exports `
  --label-config data/schema_training_trial_20261011_v001/schema/pneumonia_config.xml `
  --output-root output/schema_training_trial_20261011_v001 --run-id run_002 `
  --base-model pretrained_models/bert_base_chinese --epochs 12 --batch-size 4
```

只做转换与无模型预检时加`--check-only`。三集合按患者与完全相同全文连接分组；近重复检查单独登记，不能由精确去重推断已完成。

后续[AI样例裁决与冻结](../double_annotation_adjudication/ai-sample-adjudication.md)提供`--ai-sample-reference`及`--preserve-splits-from`入口：每例一份AI样例，严格保持旧病例/患者/全文划分，未决点屏蔽所在病例实体损失。样例监督、原始双标视图和最终专家参考分别记录；新输出使用新的run_id。

```powershell
& .\modernbert_ml_backend\.venv\Scripts\python.exe modernbert_ml_backend/predict_schema.py `
  --checkpoint output/schema_training_trial_20261011_v001/training/run_001 `
  --tasks <仅包含task_id和text的JSON数组> --output <新预测JSON文件>
```

预测入口主动丢弃人工实体、属性、关系和病例答案，仅读取任务ID和全文。输出为规范化JSON，不自动回写Label Studio，也不启动服务；无关系监督的checkpoint输出关系停用，有正例监督的checkpoint输出需复核的类型候选，分数仅表示候选端点条件下的类型分布。

## 验证与本轮边界

本轮在指定GPU环境实际完成中文预训练权重训练、保存、重载、测试预测和单次全文耗时测量；成果与数值在`output/schema_training_trial_20261011_v001/`，不复制病例或运行指标到维护文档。数据准备测试使用虚构样本，当前157项离线测试通过。实际运行校验覆盖全文末尾、推理不读取人工答案、只给显式正例计算关系损失，以及关系头保存权重确已更新。

未完成：医学最终裁决、项目实际XML确认、近重复审计、关系的完整正负参考与独立评分、无正例关系类型、服务路由/并发、Label Studio回写/UI、外部通用大模型对照。本轮报告属于模型设计和实际训练后的试验进展，不是医学效果验收。
