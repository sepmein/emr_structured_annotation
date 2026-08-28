# 试标反馈处理协议

## 输入检查

确认反馈包含运行号、指南版本、XML哈希、样本完成数和唯一问题号。身份信息泄漏或格式错误先作为数据问题处理，不进入内容裁决。

## 每条问题的必答字段

```json
{
  "issue_id": "ISS-001",
  "disposition": "clarify",
  "rationale": "为什么接受、澄清、拒绝、延期或判为数据问题",
  "decision_impact": "对病例纳入/排除/优先级/解释的影响",
  "complexity_cost": "none|low|medium|high",
  "guide_change": "具体章节和新规则；无修改时说明原因",
  "schema_change": "add|merge|rename|remove|none",
  "back_annotation_required": false,
  "validation_case_types": ["下一轮必须覆盖的案例类型"]
}
```

处置含义：

- `accept`：反馈证明现规则错误，需要改变规则或Schema。
- `clarify`：标签不变，但定义、边界、示例或默认值不足。
- `reject`：与项目目标或显式证据原则冲突；必须给可检验理由。
- `defer`：当前版本不解决，说明触发重新评估的条件。
- `data_issue`：问题来自空文本、重复、错位、泄露等数据质量，不靠标签解决。

## 跨轮追踪

相同问题再次出现时引用原问题号和上次处置。若澄清后仍在多个任务出现，不能重复写“clarify”；应重新评估定义、UI或标签设计。

指南开发者不得删除试标员的原始反馈。修订输出与原反馈并存，便于回溯。
