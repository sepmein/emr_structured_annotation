# 模型服务预测与测速

更新：2026-10-08。已实现[离线可测的测速客户端](../scripts/benchmark_model_service.py)，对接仓库当前ModernBERT `/predict` 接口。已用虚构数据、模拟响应和本机临时HTTP接口验证；尚未连接真实模型服务，没有真实模型速度或准确率结果。

该工具同时产生逐标签评分所需的模型预测与逐次测速记录。它不启动训练、不修改现有模型代码，也不证明服务已加载某个权重；实际运行前必须核实训练产物、服务配置和日志。

负责人提出新增专用模型与通用大模型的长度—耗时分布对照及医院容量估算，见[图形及测量方案](model-speed-capacity-comparison.md)。当前客户端提供其中一侧的单并发记录；配对记录制图及显式假设的容量情景计算已实现，[大模型适配入口](chat-model-benchmark.md)已通过模拟及临时HTTP验证。指定真实服务联调、并发承载及真实医院容量仍待补齐，不能把现有吞吐直接当作医院承载上限。

## 测量方式

每次请求只提交一条任务，字段为任务ID与原文`data.text`，附冻结XML和显式项目标识。不会把人工实体、双人标注、病例结论或患者字段提交为模型提示。每条任务必须返回恰好一份预测，不能从多候选中挑选有利结果。

正式计时从请求序列化开始，到完整响应解析及实体结构校验结束，包含服务处理、网络和排队。吞吐以正式阶段整段墙钟时间为分母，包括失败请求及计时记录开销。输入文件读写、预先校验及预热不计入正式阶段。当前为单并发、每请求一条任务，没有批量吞吐或压力测试功能。

每条任务仅保留**首轮正式预测**用于质量评价，后续重复请求只服务于测速与稳定性检查。首次失败仍作为失败进入逐标签分母，即使后来成功也不会替换。没有自动重试；客户端超时不证明服务器停止处理，因此超时后的服务器负载状态需另行观察。

HTTP成功还需通过实体格式校验：明确的结果数组、XML内的控件和标签、正确文本目标、合法字符偏移及原文一致的实体文本。空结果数组是合法的无实体预测；缺失结果、多任务结果、多候选、未知标签或错误偏移均记为失败。重复预测实体原样保留，由评分工具计为额外误报。关系与Choices可记录为未评价输出；本客户端不验证其质量。

当前接口返回完整NER+RE请求，测得的是该部署接口的耗时，不是只运行实体头的时间。接口不返回原文token数、实际最大长度和截断覆盖，测速报告以`null`与“未知”明确保留，不能从最后一个预测实体的位置推断文本已完整处理。

## 运行前记录

| 需要核实的内容 | 记录位置或方法 |
|---|---|
| 训练前/后实际任务模型 | 保存权重产物校验值、标签映射和训练来源；确认日志显示加载指定权重 |
| 回退与随机初始化 | 若服务未加载已知任务模型，不把随机任务头包装为可用训练前模型 |
| 服务硬件 | CPU、GPU、内存/显存、设备实际使用情况；客户端环境不能替代服务器环境 |
| 推理条件 | 实际最大长度、NER/RE阈值、关系距离设置、批量与并发、模型精度及缓存状态 |
| 文本覆盖 | 如需token与截断率，另用同一版本tokenizer/服务日志统计，保存丢失参考证据情况 |
| 项目与配置 | 正确项目标识、冻结XML、单独的测试服务进程，避免其他项目/训练改变运行状态 |
| 版本冻结 | 测试期间不更新权重；报告内返回版本一致也不能单独证明权重一致 |

这些是正式模型结果的证据要求，不是本轮已完成项。训练前任务模型与判别目标仍待负责人明确。

## 运行示例

模型服务按已有独立环境启动，入口保持：

```powershell
python modernbert_ml_backend/_wsgi.py --port 9090
```

先取得[经裁决和患者分组的测试参考](label-studio-evaluation-preparation.md)，并核实服务实际运行的模型。以下路径和模型标识是待填写的例子，当前没有对应真实预测产物：

```powershell
python scripts/benchmark_model_service.py `
  --tasks output/evaluation/splits/test.json `
  --label-config label_studio/pneumonia_config.xml `
  --endpoint http://127.0.0.1:9090/predict `
  --project PROJECT_ID `
  --run-name before --model-artifact-id VERIFIED_BEFORE_ARTIFACT_SHA256 `
  --output output/evaluation/before_run_001/predictions.json `
  --benchmark output/evaluation/before_run_001/benchmark.json `
  --timeout 60 --repeats 5 --warmup 1 --dataset-kind human_test_set
```

确认服务已切换并加载训练后模型，再用同一测试文件和参数运行，将`--run-name`改为`after`、产物标识改为实际训练后校验值，输出到新的`after_run_001`目录。工具拒绝覆盖已有测量或输入；重测使用新的运行目录，保留失败批次。

`--model-artifact-id`是操作者声明，不会自动从服务验证权重。`human_test_set`也是操作者对来源的声明，不认证医学裁决。服务需基础认证时，从环境变量`EMR_BENCHMARK_BASIC_USER`和`EMR_BENCHMARK_BASIC_PASSWORD`读取，凭据不要写进命令参数、输出文件或汇报材料。客户端不会自动读取项目`.env`。

## 接入三方逐标签评分

取得[正则预测](regex-baseline-comparison.md)和上述两版模型预测后执行：

```powershell
python scripts/evaluate_entity_predictions.py `
  --reference output/evaluation/splits/test.json `
  --regex output/evaluation/regex.json `
  --before output/evaluation/before_run_001/predictions.json `
  --after output/evaluation/after_run_001/predictions.json `
  --label-config label_studio/pneumonia_config.xml `
  --output output/evaluation/entity_comparison.json `
  --markdown output/evaluation/entity_comparison.md `
  --dataset-kind human_test_set
```

预测文件与评分格式直接兼容，失败任务保留原文与`status=failed`、空实体数组；评分仍计其漏检，失败阴性也不获得辅助准确率的正确分数。

## 汇报速度表

| 结果 | 训练前模型 | 训练后模型 | 取值与口径 |
|---|---|---|---|
| 独立测试任务数 | 待测 | 待测 | `unique_tasks`，不是重复请求数 |
| 正式请求数 | 待测 | 待测 | `measured_task_runs` |
| 成功/失败/失败率 | 待测 | 待测 | `successful_task_runs`、`failed_task_runs`、`failure_rate` |
| 成功延迟P50/P95 | 待测 | 待测 | `p50_success_latency_ms`、`p95_success_latency_ms` |
| 有效吞吐 | 待测 | 待测 | `successful_tasks_per_second`，失败耗时留在墙钟分母 |
| 预热失败数 | 待测 | 待测 | `warmup_failed_runs`，与正式失败分开 |
| 运行稳定性 | 待测 | 待测 | 返回版本计数及后续结果与首次不同的次数 |
| 字符长度分布 | 待测 | 待测 | `text_chars`，含P50/P95与最大值 |
| token数与截断覆盖 | 当前接口不提供 | 当前接口不提供 | 另附同版本tokenizer或服务日志，不能填成0 |

全部失败时成功延迟为`null`，吞吐为0；不能以失败请求耗时冒充成功速度。多个返回版本或重复预测发生变化时，先核实原因，不能直接合并为稳定的某版性能。没有返回模型版本的情况会标记`unreported`。

报告还保存逐次样本、预热样本、输入和工具SHA256、操作者声明的模型产物标识、运行时间、客户端环境。原始响应与错误正文不写入测速报告，避免复制病历或凭据；失败仅保留类别和HTTP状态。预测文件含病历原文，应保存在受控本地`output/`目录。

正则工具目前测本地CPU抽取，不能与上述HTTP端到端结果直接计算提速倍数。可以在汇报中分别呈现计时层，并补充相同层的公平对照；整个NER+RE接口与规则实体抽取的功能范围差异也须说明。

## 本轮验证范围

[测试代码](../tests/test_benchmark_model_service.py)覆盖：不提交参考答案、预热与正式分离、首次失败保留、失败分母、全失败、结果歧义、偏移/标签/控件错误、重复实体、版本及预测变化、输出覆盖保护。临时本机HTTP接口验证实际JSON请求、正常响应、HTTP错误、非法JSON和重定向拒绝；模拟超时验证计数逻辑。没有加载模型，也没有调用真实部署服务。
