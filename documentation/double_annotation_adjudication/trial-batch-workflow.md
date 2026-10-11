# 多项目试标导出整合、统计与裁决准备

实现入口：[trial_batch.py](../../emr_annotation/annotation_analysis/trial_batch.py)。它复用现有Schema解析、标注解析、对照、指标与V2渲染，不自动作医学裁决或回写Label Studio。

## 输入与整合合同

使用明确的原始JSON目录，只遍历其直接文件，不沿兼容Junction重复清点。每个文件必须是单项目任务数组。任务键为项目ID和Label Studio任务ID，业务编号仅作辅助显示。

多个导出中的同一任务，只有`data`和标注ID、实际人员、取消状态、已提交`result`相同时才纳入一次。草稿和其他导出元数据差异写入来源映射。相同提交选择文件名时间较晚的容器，不选择“较新答案”。原文或已提交答案冲突、缺双标、同人多版本、缺文本进入阻断清单，不自动采纳。相同文本的不同任务只标记重复组，不自动删除。

每个任务按两名实际`completed_by`配对，组内A/B映射固定。一个项目出现多组人员时分别计算κ，保留实际项目ID、任务ID和账号，禁止把第三个账号改名映射。角色未确认的账号显示待核实。

## 配置与命令

当前流程使用明确声明的候选XML，快照记录SHA256，并在全部新页面提示尚需核实实际项目配置。候选文件兼容不证明项目标注时的控件、属性作用域或保存行为已经确认。配置不同的合同应分开运行。

```powershell
python -m emr_annotation.annotation_analysis.trial_batch --input-dir data/trial_annotation_exports_20261010_v001 --output-root output/trial_annotation_exports_20261010_v001 --run-id run_20261010_03 --candidate-dir data/annotation_schema_candidates_7projects_20261010_v001/schema_candidates --fallback-schema data/annotation_schema_candidates_7projects_20261010_v001/schema_candidates/project-220.candidate.xml
```

无模型模块允许`python -m`，不改变ModernBERT服务启动合同。后续处理改用新run_id，禁止覆盖旧运行。若候选目录没有某项目文件，使用显式fallback，并在该项目快照清单记录来源及未确认状态。不能根据文件名宣称已取得真实项目配置。

## 产物位置

| stage | 文件 | 用途 |
|---|---|---|
| data_preparation | merged_tasks.json、source_lineage.json/CSV、blocked_tasks.csv、schema_candidates/ | 整合原始内容、全部来源版本、阻断记录及候选快照 |
| annotation_analysis | statistics.html、statistics.json、统计说明.md | 动态统计图表、可复算明细及分母说明 |
| annotation_analysis | tasks、annotations、entities、attributes、relations、issues CSV | 逐记录追溯，属性表只含可比较的匹配槽位 |
| annotation_analysis | project_agreement、pair_agreement、label_agreement、project_label_agreement CSV | 项目和人员组合、逐标签严格一致性及覆盖 |
| annotation_analysis | attribute_agreement、pair_attribute_agreement、choice_value_distribution CSV | 属性一致性、全部适用实体的填值覆盖、原始选项分布 |
| annotation_analysis | relation_agreement、project_relation_agreement、pair_relation_agreement CSV | 已记录类型的观察数量、子集指标和全量不可评价状态 |
| adjudication | review.html、audit.json、核查报告.md、review_queue.csv、case_disagreements.csv | V2人工裁决工作台及定位队列 |
| adjudication | 裁决模板.json、标签核查签署模板.json、使用说明.md | 逐项最终答案和49标签阅读全文查漏签署 |
| delivery | index.html、manifest.json | 总入口、源文件/代码/规则/输出哈希及验证凭据 |

所有产物均在`output/<输入batch_id>/<stage>/<run_id>/`。原始数据和包含原文的整合JSON保持受控访问，统计页面不嵌入患者/就诊直接标识值；显示病历片段的工作台仍不构成完整去标识化验收。

## 统计口径

- 病例一致率只用有效单选配对，κ按固定人员组报告，单例不解释。汇总不提供混合人员的κ。
- 实体使用任务、start、end、标签的多重集合精确匹配，严格对称F1为`2×共同/(A+B)`。错误和重复保持在原始描述性统计中。未覆盖标签不计100%。
- 属性只比较双方各自唯一的严格匹配实体，并按候选Schema作用域核定。非空一致率为非空相同/至少一方填写，单方空白不从分母移除，双方空白单列。每组属性κ仅使用双方有效单选子集，分母另列；全批不混合κ。
- 属性填值覆盖按适用实体标签槽位计数，多标签选区可能产生多个槽位。填值存在不证明取值或作用域正确。未匹配实体的属性仍需医学复核。
- 关系比较实际端点、方向及类型，不推断缺失类型。类型或端点记录不完整时全量类型F1不可评价；已记录类型子集的观察数不能代替全量质量。端点类别是否合法另列结构候选。
- 同一选区可触发多个问题，问题条数、核查条数与独立错误数不等价。每例增加全文回读项，双方同错不能凭一致指标关闭。

## 人工处理与验证边界

先确认配置、人员身份和源版本，再修复偏移与引用等技术问题。医学裁决按病例、49标签全文查漏、9属性、5关系推进，分别留下答案、证据偏移、规则、理由、裁决人和日期。最终标注副本、裁决凭据和原始双标指标分别保存。

V2页面按批次来源/规则绑定浏览器草稿，导出仍是裁决记录JSON。统计矩阵可定位病例，热力格可定位项目标签；同页面的哈希变化也会重新定位。不能强行导入另一批答案。当前系统没有多人同步或自动金标转换。

离线验证使用`python -m unittest discover -s tests -v`。浏览器需另测筛选、定位、关系图、导出/导入和响应式布局。结构和UI验证不证明医学答案正确，也不证明Label Studio真实项目保存行为已验证。
