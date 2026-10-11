# 人工审查后的双标一致性与符合度统计

入口为 [human_reference_statistics.py](../../emr_annotation/annotation_analysis/human_reference_statistics.py)。本工具只读原双标病例快照和冻结参考，不修改审查数据库、原始导出或裁决答案。

两类指标分别计算：双人一致性比较同一病例的两份原始标注，符合度将各人的原始标注与该病例的人工冻结参考比较。人员按原始账号和实际病例集合统计，不将各项目的A/B角色合并为两名固定人员，不用于人员能力排名。

```powershell
python -m emr_annotation.annotation_analysis.human_reference_statistics `
  --review-dir output/<batch_id>/adjudication/<review_run> `
  --freeze-dir output/<batch_id>/adjudication/<freeze_run> `
  --model-predictions output/<batch_id>/delivery/<prediction_run>/real_case_predictions.json `
  --test-data output/<batch_id>/training_data/<data_run>/test.json `
  --output-dir output/<batch_id>/annotation_analysis/<new_run_id>
```

输出 `statistics.json`、`per_annotation_scores.json` 与 `examples_pool.json`。输出目录必须不存在，重新运行使用新run_id。记录输入及代码哈希、Git版本和未提交状态。真实病历节选与PPT构建数据只保留在output中。

实体按原文起点、终点和标签严格匹配，任务ID隔离，重复实体按多重集保留。双标实体一致F1为 `2 × 相同实体数 / 双方实体总数`。与参考比较的精确率、召回率和F1分别保留分母。完全漏标是参考实体与原标任何实体都没有重叠跨度，不能将边界或标签差异计作完全漏标。

病例四分类报告原始一致率和各实际人员配对的Cohen κ。属性一致率仅在唯一、严格匹配且该字段适用的实体上计算，排除双方均空白，一方空白计为不一致。关系原导出存在缺型或端点歧义时，不据此推断完整关系质量。

AI裁决草稿是人工参考的起点，与最终参考有依赖，符合度不能当作独立测试准确率。训练模型指标与案例只使用真实测试病例，合成数据不进入该统计。人工参考未经独立专家签署时应如实保留来源状态。

PPT在独立delivery运行目录中生成，保留来源数据、逐页原文定位、可编辑图表工作簿、统计复算凭据、构建脚本及逐页预览。文件包和版式检查不等同于原生PowerPoint验收。

## 每个实体标签的人工错分类统计

入口为 [entity_label_errors.py](../../emr_annotation/annotation_analysis/entity_label_errors.py)，仅使用原始人工标注与冻结参考，不读取AI预测。

```powershell
python -m emr_annotation.annotation_analysis.entity_label_errors `
  --review-dir output/<batch_id>/adjudication/<review_run> `
  --freeze-dir output/<batch_id>/adjudication/<freeze_run> `
  --output-dir output/<batch_id>/annotation_analysis/<new_run_id>
```

输出 `entity_label_statistics.json`、`label_profiles.csv`、完整Schema标签间的 `wrong_label_matrix.csv` 和可按病例、标注ID、原文位置追溯的 `error_ledger.json`。输出目录必须不存在，输入冻结清单中的文件哈希先逐一核对。

错分类流向从原标标签指向参考标签，仅在起止位置完全相同且参考类别唯一时计数。每个未匹配的原标实体只进入一个错误类别；重复的正确类别实体不算错分类，多种参考类别不强行选方向。边界偏差、类别与边界混合偏差、无重叠多标另列。逐标签错分类率的分母是该原始标签的标注次数。

参考侧的完全漏标及其他缺失另行按参考实体计数，不与原标侧错分类数相加。每例两份原标分别对照参考，因此合计参考人次为参考实体数乘两份原标；同一处原文可能对应两次比较。所有Schema标签都保留，无原标时错类率显示为不可计算，无参考实体时F1不作为可评价值。少量样本的比例应同时展示次数和分母。
