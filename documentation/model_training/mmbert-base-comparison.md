# mmBERT 与中文 BERT 的基座替换对比

本入口复用[Schema多任务试验](schema-multitask-trial.md)，对照为已有中文BERT的AI样例训练运行，不包含原版ModernBERT。基础模型采用官方[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)，本地目录为`pretrained_models/mmbert_base/`；revision、下载方式与官方LFS SHA256核对保存在该目录的`download_receipt.json`。数据不上传外部服务。

## 比较合同

沿用同一批AI样例裁决参考与患者划分，训练/验证/测试、完整参考、Schema JSON和XML必须字节一致。使用相同实体跨度头、属性头、病例头、关系头、优化器学习率、batch、seed、阈值、窗口长度、重叠、跨度宽度、轮次上限与早停规则；只在验证集选轮次。

两种编码器均微调最后4层，其余编码器参数冻结；中文BERT为12层，mmBERT为22层。官方分词器随基座更换，导致窗口数量、实体可表示性、有效监督窗口和每轮更新次数可能不同。参数量也不同，本轮不是等容量或等更新次数的纯架构消融。没有同时引入字符边界修复、规则融合或测试调参。关系只评价已知正例召回，属性只评价已知槽位；AI样例结果不等于专家医学金标。

## 保存与推理

共享网络的`unfreeze_encoder_tail`适配两类编码器层布局，BERT的原冻结策略保持。mmBERT保存`frozen_encoder_delta_v1`：仅保存更新的编码器参数、完整任务头与必要缓冲区，避免重复存储冻结的基座。重载核对基座配置与权重SHA256，合并更新后严格加载完整状态并检查有限值；缺少基座或哈希不符时失败。不要把这个checkpoint作为独立完整编码器权重使用。

`predict_schema.py`兼容已有完整state dict及上述增量格式，继续只从输入提取task_id和text，不读取人工答案。基础权重需保留，移动工作区时要同步恢复并校验；参数路径仍采用脚本命名空间。

## 复现命令

从仓库根目录使用独立GPU环境。后续运行更换run_id，不覆盖既有产物：

```powershell
& .\modernbert_ml_backend\.venv\Scripts\python.exe modernbert_ml_backend/train_schema.py `
  --ai-sample-reference output/schema_training_trial_20261011_v001/adjudication/run_001/proposal_pass_004/proposals.json `
  --preserve-splits-from output/schema_training_trial_20261011_v001/training_data/run_003 `
  --label-config data/schema_training_trial_20261011_v001/schema/pneumonia_config.xml `
  --output-root output/schema_training_trial_20261011_v001 --run-id run_mmbert_next `
  --base-model pretrained_models/mmbert_base --epochs 12 --batch-size 4
```

本次`run_mmbert_001`保留完整保存失败回执，原因是E盘剩余空间不足以容纳完整权重；`run_mmbert_002`另行采用冻结基座增量保存。实际启动脚本、训练时源码哈希快照、合成架构检查与checkpoint往返检查均归入本批次output，不改写历史结果。

对比生成器是[base_model_comparison.py](../../emr_annotation/evaluation/base_model_comparison.py)，无模型导入或联网副作用；它核对冻结输入、公共参数、保存重载状态、预测范围与原文，并独立重算实体分数。其他任务使用相同训练评分器的已保存结果。输出六部分报告、49类实体附表、已审阅的数据快照和比较JSON。示例：

```powershell
python -m emr_annotation.evaluation.base_model_comparison `
  --baseline-data output/schema_training_trial_20261011_v001/training_data/run_003 `
  --baseline-training output/schema_training_trial_20261011_v001/training/run_003 `
  --candidate-data output/schema_training_trial_20261011_v001/training_data/run_mmbert_002 `
  --candidate-training output/schema_training_trial_20261011_v001/training/run_mmbert_002 `
  --output output/schema_training_trial_20261011_v001/delivery/run_mmbert_next
```

## 验证边界

合成tiny编码器检查只验证层冻结和实际反向传播，不作为原版ModernBERT效果对照。增量格式用合成模型验证更新后所有重建张量精确一致、错误基座拒绝及缺少更新拒绝；真实模型还需要实际训练、保存重载、只给原文的独立推理与评分核查。报告的本机单次耗时不代表稳定服务或并发性能；独立专家测试、多随机种子和近重复审计另行开展。
