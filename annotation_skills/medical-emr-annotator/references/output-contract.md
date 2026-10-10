# 反馈输出契约

`annotator_feedback.json` 必须是UTF-8 JSON对象：

```json
{
  "run_id": "pilot-001",
  "guide_version": "0.1.0",
  "xml_sha256": "...",
  "sample": {"total": 10, "completed": 8, "blocked": 2},
  "issues": [
    {
      "issue_id": "ISS-001",
      "category": "boundary",
      "severity": "major",
      "affected_tasks": ["T003", "T008"],
      "affected_labels": ["肺炎诊断"],
      "evidence_excerpt": "去标识化的最小必要片段",
      "observed_problem": "两种可执行解释是什么",
      "question": "希望指南明确回答的问题",
      "decision_impact": "可能改变纳入/排除/复核优先级/证据解释，或无直接影响",
      "temporary_action": "blocked或本轮采用的保守处理",
      "suggested_options": ["选项A", "选项B"]
    }
  ]
}
```

要求：

- `issue_id` 唯一；`affected_tasks`只用试标任务号。
- `temporary_action`只能描述本轮怎样停下或保守处理，不得宣布新规则。
- `decision_impact`必须明确关联最终病例判断；纯界面建议可写“无直接病例判断影响”。
- 无问题时保留空数组，不虚构问题。

`annotation_results.json` 至少包含 `run_id`、`guide_version`、`tasks[]`。每个任务包含任务号、实体/属性/关系、病例级判断、最小证据和 `issue_refs`。

`run_summary.md` 至少报告完成率、阻塞问题、每类问题数量、最常见问题、是否建议进入下一轮扩大试标。
