# mmBERT优化试验与小模型病例分类

目标分别为严格实体micro F1与病例四分类macro F1，争取80%—85%以上。当前本批次未达标，保留实际结果和未完成验证，不按测试集改阈值或改选分支。原版ModernBERT不参与对比。

2026-10-12 用户更新执行重点为[逐项错分原因分析与改进](error-driven-improvement.md)：先建立可追溯错误清单，再按主要原因实施和验证；新一轮查找与执行改进方案在北京时间00:50及之后启动。

冻结182例人工参考及1000条规则合成样本沿用[人工裁决后混合训练](human-reviewed-hybrid.md)。抽取训练使用1128例；病例小模型比较真实128例单独训练及合成低权重混合。验证、测试各27例真实病例，历史上已用于开发，不是独立外部验收集。人工答案和原始导出保持不变。

新增代码入口：`train_small_case.py`、`evaluate_modular.py`、`predict_modular.py`，均在`modernbert_ml_backend/`，ML实现进入`training/`。模型重载必须保留配置引用的抽取器、分词器、基座和独立头，不能只搬一个权重文件。E盘output Junction对应D盘真实目标必须一并备份。

## 实施内容

抽取器增加token包围跨度＋字符边界修正、宽度偏置和同标签重叠抑制，保留局部可表示实体监督；调整编码器/头学习率、难负例权重、真实/合成权重，加入累积、预热及调度。第一轮验证87.12%、测试78.75%，第二轮验证88.68%、测试76.44%。第二轮因验证更高而选中，不能因测试更低改回第一轮并宣称严格选模。

分层病例头消费症候群/检出预测输出，按信息不足、是否目标相关、目标/复核三项分解保留四类概率。试验未改善测试macro F1：验证选中头测试59.00%，原融合头67.42%。病例LR/XGBoost比较文本、文本向量、证据及组合；表内个别测试分支更高，属于探索观察，不据此改选。

结果、分支比较与六部分网页报告在`output/schema_training_trial_20261011_v001/delivery/run_optimization_001/`；选中组件及评估为`evaluation/run_modular_002/`。病例/辅助模型仍引用`training/run_hierarchical_001`与任务对应抽取器，NER组件为`training/run_refined_002`。没有替换现有Label Studio在线服务。

```powershell
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/train_small_case.py --help
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/evaluate_modular.py --help
& modernbert_ml_backend/.venv/Scripts/python.exe modernbert_ml_backend/predict_modular.py --checkpoint output/schema_training_trial_20261011_v001/evaluation/run_modular_002 --tasks <task_id和text的JSON> --output <新预测JSON>
```

独立GPU环境本轮增加XGBoost 3.1.1，scikit-learn原已存在；版本写入小模型回执，不合并根环境和原服务requirements。模块不会在CLI帮助阶段加载ML库或权重，业务训练模块有ML依赖。用于病例训练的预测证据目前是训练样本本身的抽取结果，未执行折外抽取；独立原文TF-IDF不依赖实体预测。

## 属性与后续设计

9个perRegion属性都在Schema多任务抽取层训练/输出。病原三检测属性逐值审计与联合记录已增加，不能用大多数“原文未说明”带来的高整体正确比例宣称具体标本有效。标本/检测方法尚未显式进入融合证据向量；Laya式实体条件字段查询、完整检测记录特征、独立校准及新增事件Schema未实施，分析和文献见[Laya架构核查](laya-architecture-analysis.md)。

后续优先补充真实目标/复核病例、具体标本方法和关系负例，执行患者分组折外抽取，在新的独立批次上验收。辅助证据分组与病原检出没有独立专家金标；检出不等于病因，真实关系无完整负例时不报告完整P/F1。
