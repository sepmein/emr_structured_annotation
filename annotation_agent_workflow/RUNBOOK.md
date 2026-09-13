# 双角色运行手册

## 1. 开发或修订指南

调用：

```text
Use $medical-annotation-guide-developer.
读取当前成人/儿童XML、后端标签提示、现有指南和上一轮annotator_feedback.json。
所有输出写入本轮developer目录；XML只作为当前事实来源，标签变化生成提案，不直接修改。
```

首次运行没有反馈时，要求生成 `annotation_guide_v0.1.0.md`、空反馈处置文件、Schema变更提案、决策日志和试标计划。

## 2. 使用指南试标

调用：

```text
Use $medical-emr-annotator.
严格依据指定版本指南和当前XML，对本轮去标识化病例试标。
所有输出写入本轮annotator目录；不要修改指南或XML；无法判断时记录结构化问题。
```

完成后必须运行该Skill中的 `scripts/validate_feedback.py` 校验反馈JSON。

## 3. 回交反馈

再次调用指南开发者，输入原指南、`annotation_results.json`、`annotator_feedback.json`和`run_summary.md`。要求逐条生成反馈处置，不能覆盖试标员原始文件。

## 4. 负责人审批点

以下变化必须由项目负责人确认后才能进入XML：

- 目标病例的纳入/排除定义；
- 新增、删除、合并或重命名标签；
- 关系语义变化；
- 需要回标既有数据的规则变化；
- 从人工标注迁移到自动推断的字段。

## 5. 推荐目录

```text
annotation_agent_workflow/
  guides/
  runs/
    round-01/
      inputs/
      annotator/
      developer/
```

每轮在输入清单中记录指南版本和XML SHA-256，确保问题可以复现。
