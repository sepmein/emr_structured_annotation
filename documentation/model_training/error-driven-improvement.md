# 按错分原因改进模型：当前执行目标

更新时间：2026-10-12，Asia/Shanghai。用户要求更新目标，先逐项分析当前 F1 不高的原因，再提升模型准确性；查找和实施改进方案在 **2026-10-12 00:50 及之后** 开始。

本页补充并约束本线程已有的 [mmBERT 优化目标](mmbert-optimization.md)，不将未完成的性能目标标记为完成。临时等待已结束：本轮在北京时间2026-10-12 00:50之后收到一次性续跑授权并执行，没有创建新的周期任务。

## 更新后的目标

基于已冻结的人工裁决参考和现有训练数据，逐项核对实体抽取、实体关联属性、关系及病例分类的错分，输出可追溯错误清单和主要原因统计。以确认的主要错误为依据，查找、实施并通过对照实验验证改进；分别争取严格实体 micro F1 与病例四分类 macro F1 达到 80%—85%以上。完成实际训练、权重重载、分支比较和更新报告，并如实记录未达标部分。

重点先建立错误证据，再决定是否更换基座、设计任务头或调整训练参数。不得将推测写成已验证原因。

## 逐项分析口径

- 实体：对每个误报、漏报和错标记录病例/实体标识、原文位置、参考答案、模型答案及对应错误；区分跨度边界、标签混淆、重复/重叠、上下文与长文本覆盖等可核查问题。
- 属性：逐条核对主体、存在状态、时间、条件/假设和病程信息，特别核对检验阳性/阴性、标本类型、检测方式与具体病原体的关联；分别报告各属性值和完整记录，避免多数值掩盖少数值失败。
- 病例：对每个错分病例保留参考类别、预测类别、置信度和原文证据，结合混淆矩阵分析是上游抽取错误、证据组合、类别样本不足还是分类决策问题。原因需要对照或消融验证。
- 关系：区分端点抽取错误与关系类型问题。没有完整负例参考时，保留可核查案例和适用的正例指标，不伪造完整精确率或 F1。

逐项清单由工具批量生成；提供人类易读的审查视图和汇总，避免在对话中逐条展开所有原文。

## 改进与验证约束

1. 原人工裁决、原始导出和历史训练/评估结果保持不变。仅提出需要人工再次核查的疑点，不自行改参考答案以提升分数。
2. 沿用患者分组，合成数据只进入训练；真实验证/测试不混入合成数据。已使用的本批测试集可做开发诊断，不能再声称独立验收。
3. 新方案的训练、样本选择、阈值和选模依据训练/验证或患者分组交叉验证。记录开发测试参与诊断的范围；最终验收仍需未参与开发的新真实批次。
4. 先形成“错误证据—改进假设—单项对照—实际效果”的可审查链路；记录改善了哪些错误、是否引入新错误，以及各类 F1、误报、漏报和属性能力的变化。
5. 核查官方 mmBERT-large 可用性后再决定是否采用。继续比较适合的任务头和小模型，不与原版 ModernBERT 对比；不将模型变大当作效果保证。
6. 复用当前 Schema、转换、评分和训练入口，新代码归入现有业务包。保留当前未提交改动、两套环境、现有线上服务和模型权重；新运行使用新的 run_id，不覆盖旧结果。

## 执行入口与交付

启动时先读根 AGENTS.md、项目组织规则、[当前优化说明](mmbert-optimization.md)和[属性架构分析](laya-architecture-analysis.md)。
复用现有 `emr_annotation/annotation_analysis/entity_label_errors.py`、`emr_annotation/evaluation/pathogen_attribute_audit.py`、评分模块及 `modernbert_ml_backend/` 的模块评估和训练入口。

本轮既有证据入口为 `output/schema_training_trial_20261011_v001/evaluation/run_modular_002/` 和同批次 `delivery/run_optimization_001/`；人工冻结和患者划分沿用上游清单。新诊断、训练与报告仍留在同批次对应 stage 的新 run_id 下。

后续必须交付错误清单、主要原因统计、可读案例、按原因选择的改进方案、实际运行/重载凭据、前后对照报告。只有实测满足目标才可标记目标完成；否则明确下一项可执行工作或实际数据缺口。

## 2026-10-12 本轮实测与目标状态

目标未完成。组合实体方案在本批开发诊断达到严格micro F1 **82.55%**（原76.44%）；病例四分类仍保留验证选中的原分层融合，macro F1 **59.00%**。新增聚焦文本LR为65.42%、临床线索树为64.50%，但验证F1未超过原0.60，因此没有按开发测试最高值替换分类器。

错误账本改进前397条、改进后336条，分别包含实体、属性、病例和已知关系未命中；这些分母不同，不能相加为独立病例。95/113个实体误报是数值/体温、35/88个漏报是时间；96/201个实体错误来自同一病例。阳性误判6处由同记录原文绑定修复，阳性1/7→7/7（3位患者）。具体标本仅1处真实参考且未识别，不能根据多数“原文未说明”验收多标本能力。

实际训练的局部候选头按验证选阈值；候选补全和编号/原文约束组合后误报113→84、漏报88→64。实体查询注意力属性头训练20轮未超过验证基线，最佳保留epoch 0，不能声称该头取得了改善。编号约束还删除了7个参考实体，疑点单独保留，原人工参考四个文件哈希本次全部一致。

本轮路径（同批次、新run_id）：

- `evaluation/run_error_audit_test_001/`：原模型错误清单。
- `training/run_context_001/`：实际训练局部候选/属性及全文病例头。
- `training/run_attribute_context_001/`：实体查询注意力属性头，验证未接受。
- `training/run_explicit_evidence_001/`、`training/run_combined_evidence_001/`：按验证校准文字证据与组合方案。
- `training/run_focused_case_001/`、`training/run_clinical_cue_trees_002/`：患者3折OOF与真实验证选模的病例对照。树模型第一轮的并行概率未满足逐位重载比较，失败回执保留，第二轮确定性执行通过。
- `evaluation/run_combined_reload_001/`：真正从基础/增量权重及任务头重建预测；27例实体、关系、病例类别与概率和保存结果完全一致。推理只读task_id/text。
- `delivery/run_error_driven_002/`：六部分报告、源证据、前后完整网页错误清单、原文参考疑点、204项离线测试日志和浏览器检查。

E盘output保留对应Windows Junction；上述新增产物实际位于 `D:/codex_outputs/emr_structured_annotation/schema_training_trial_20261011_v001/`。需同时保留D盘目标和原基础权重，不能仅复制联接。报告本地预览：`http://127.0.0.1:4190/report_app/dist/?view=1`；原人工审查服务与原报告保留。

保留的病例模型仍使用原专用抽取器和已选分类头；新实体纠错没有自动进入其训练特征。下一轮先补目标/复核真实少数类、复核参考疑点，再以按患者完整折外的纠错实体特征训练病例融合，并对合成数据来源做分层消融；最后由新真实批次独立验收。没有确认官方mmBERT-large可用权重，本轮继续base，不与原版ModernBERT比较。

## 可复现入口

只读错误账本使用根环境，不加载模型：

```powershell
python -m emr_annotation.evaluation.model_error_audit --data-dir output/schema_training_trial_20261011_v001/training_data/run_hybrid_002 --extractor-predictions output/schema_training_trial_20261011_v001/evaluation/run_modular_002/extractor_predictions.json --case-predictions output/schema_training_trial_20261011_v001/evaluation/run_modular_002/test_predictions.json --output-dir <新的evaluation运行目录>
```

真实训练/重建用已有 `modernbert_ml_backend/.venv/Scripts/python.exe`：`train_error_driven.py`、`train_attribute_context.py`、`train_focused_case.py`（可加`--trees`）、`calibrate_evidence.py`（可加`--candidate-heads`）和`predict_error_driven.py`。各CLI的`--help`给出输入参数，实际参数与源哈希见对应运行回执。不要覆盖旧run_id。

报告生成：`python -m emr_annotation.evaluation.error_driven_report --batch-dir output/schema_training_trial_20261011_v001 --output-dir <新的delivery运行目录>`。模板为`emr_annotation/evaluation/templates/error_driven_report.jsx`；Data共享运行时构建与检查回执留在交付目录。病例评分原函数已归入`emr_annotation/evaluation/case_classification.py`，旧训练模块导出同一函数，保持原口径。
