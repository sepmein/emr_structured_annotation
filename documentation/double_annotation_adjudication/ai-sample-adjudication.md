# AI样例裁决与流程试验

本入口将已有双人导出转为本地AI样例，供跑通训练与评估，不生成专家签署金标。病例结论及例外修订由AI逐病例复核；实体、属性、关系使用字面规则辅助，原始每条result登记处理及理由。全文标签覆盖为扫描/对账记录，不代表专家已全文确认完整性。

源文件不改写。具体病例说明、例外及最终副本都放`output/<batch_id>/adjudication/<run_id>/`，不写入源码/测试/规范。规范源仍为[当前指南](../annotation/annotation_guide_v2.1.1.md)、[字典](../annotation/label_dictionary_v2.1.1.md)和项目XML快照。用户明确允许本轮本地试验继续时，按授权口径记录scope_decision；不得据此伪称完成其他验收。

人工全量审查使用[格式化审查与裁决工具](human-review-tool.md)：批量显示双人原标注和AI裁决，逐病例编辑实体、属性、关系和病例分类，保存版本、恢复历史及导出人工审查结果。旧AI样例和训练冻结文件不改写。

## 使用与产物

```powershell
python -m emr_annotation.adjudication.ai_sample --tasks <整合任务JSON> --schema <项目XML> --notes <病例理由JSON> --overrides <逐点例外JSON> --output-dir <新的裁决运行目录>
```

病例理由格式为源数组序号到`[病例结论, 条款, 理由]`的映射。例外按相同序号记录实体条件、属性修订、来源标签修订及理由。运行保存源内容哈希；修订重跑新目录，不覆盖此前结果。

- `proposals.json`：每例实体、属性、关系、病例答案及`ai_sample_not_expert_gold`状态。
- `point_decisions.json`：全部原始result逐点处理与最终属性/新增点。保留、修订、删除、不沿用与待复核分别列示，不强行选择某一标注者。
- `point_summary.json`、`label_coverage.json`：处理计数和逐病例49标签扫描结果。待复核原始点屏蔽所在病例的实体损失，属性仍按合法已知槽位处理。
- `label_studio_ai_samples.json`：作为predictions导入另一受控样例项目的文件，不提交专家annotations；只完成本地Schema解析，真实界面导入需单独验收。
- `review.html`：本地逐病例、逐点查看及下载。包含原文，统计报告只列任务号/汇总。

实体负监督仅在未发现未决点的病例中暂用“样例已补齐”的假设，仍是弱监督，不证明医学完整性。关系只监督已知正例，不把未知实体对变成无关系。未能裁定的来源、简称、检验冲突保留未知或待复核。

## 冻结、训练与报告

[AI样例冻结模块](../../emr_annotation/training_data/ai_sample.py)要求沿用已有患者/相同全文划分；样例病例集合、文本和患者键必须与旧划分一致。每例一份参考，不重复计两名标注者视图。

```powershell
& .\modernbert_ml_backend\.venv\Scripts\python.exe modernbert_ml_backend/train_schema.py `
  --ai-sample-reference <proposals.json> --preserve-splits-from <旧training_data运行目录> `
  --label-config <本轮XML快照> --output-root <同batch输出目录> --run-id <新run_id> `
  --base-model pretrained_models/bert_base_chinese --epochs 12 --batch-size 4
```

模型和参数见[Schema多任务试验](../model_training/schema-multitask-trial.md)。[样例报告生成](../../emr_annotation/evaluation/ai_sample_report.py)沿用六部分框架；实体仅对未屏蔽病例评分，病例对全部有效样例评分，属性只评价已知槽位，关系报告已知正例召回而不报告完整P/F1。旧模型可对同一新参考重评分；原始双标参考下旧分数不能直接解释为新模型提升。

[真实运行校验](../../emr_annotation/evaluation/verify_ai_sample_run.py)单独执行GPU权重重载、输入输出哈希、全文尾部覆盖、推理不读取参考答案、未知关系对无负损失以及本地Label Studio预测解析。离线测试只使用虚构病历；上述检查不代替专家、真实服务/UI及独立医学盲验。
