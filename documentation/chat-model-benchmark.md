# 外部或本地大模型：结构化抽取、逐标签评价与测速入口

更新：2026-10-09。已实现[Chat Completions兼容接口客户端](../scripts/benchmark_chat_model.py)，完成模拟响应、本机临时HTTP接口和20条虚构任务的离线预检。没有调用真实外部模型或本地大模型；具体模型、服务地址和运行条件仍待确认。

此工具落实“先外部模型测试、最终本地大模型同资源对照”的两阶段安排。它读取与专用模型相同的冻结任务文件，生成实体评分器和[分布图工具](model-speed-capacity-comparison.md)可直接读取的预测/测速记录。它不导入模型依赖、不训练、不调用工具、不自动修复答案、不重试，也不执行去标识化。

## 输入与任务范围

输入为标准化数组，含`task_id`与精确原文`text`；可直接读取[完整导出转换工具](label-studio-to-training.md)生成的`test.json`。即使输入还带人工实体、属性、关系或患者字段，请求只发送原文及其字符数，系统提示使用冻结的标签/关系范围与指定标注规范，不发送本条人工答案或患者元数据。

当前要求输出NER+RE：实体标签来自XML的49类，关系来自5种配置类型；不要求属性或病例分类。实体必须给出ID、原文字符起止位置、原文跨度及标签；关系须指向已输出实体。病历原文不清洗、不截断、不分段；非BMP字符按Python字符位置计数。若服务不能接收或返回完整任务，保留失败，不自动缩短输入或改成小范围任务。

提示模板是待真实数据校准的初版，不能仅凭提供了规范宣称大模型已充分适配。使用训练/验证数据调整提示，最终测试前冻结。两侧任务范围、全文覆盖和质量需单独核实；现有ModernBERT接口同时运行NER+RE，本轮大模型也要求生成关系，当前独立质量评分只覆盖实体。关系只通过有限结构检查，不宣称关系效果已评价。

## 先离线预检

以下命令已经以20条虚构任务运行过，不读取密钥、不创建网络请求，也不加载模型。`example.invalid`和`MODEL_NOT_SELECTED`均为占位，不代表选定服务：

```powershell
python scripts/benchmark_chat_model.py `
  --tasks documentation/evaluation_examples/label_studio_export_demo/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --guidance documentation/annotation/label_dictionary_v2.1.1.md `
  --endpoint https://example.invalid/v1/chat/completions `
  --model MODEL_NOT_SELECTED --deployment external_api `
  --max-output-tokens 4096 --dataset-kind synthetic_demo `
  --check-only --output-dir output/chat_model_preflight
```

输出`run_receipt.json`及不含任务原文的冻结`prompt.txt`，不生成预测或测速文件。可查看[本轮预检回执](evaluation_examples/chat_model_preflight/run_receipt.json)。字典文件名仍为v2.1.1，内容版本以本次冻结文件及校验值为准，不能把文件名直接当作当前已验收版本。

## 接入确定的真实服务

选择支持Chat Completions协议的具体模型和完整接口地址，确认请求参数支持情况。以独立测试集为例，以下都是需要填写实际值的命令模板：

```powershell
python scripts/benchmark_chat_model.py `
  --tasks output/evaluation/training_prepared/splits/test.json `
  --label-config label_studio/pneumonia_config.xml `
  --guidance documentation/annotation/label_dictionary_v2.1.1.md `
  --endpoint https://YOUR_SERVICE_HOST/v1/chat/completions `
  --model YOUR_EXPLICIT_MODEL --deployment external_api `
  --api-key-env EMR_LLM_API_KEY --max-output-tokens 4096 `
  --token-limit-field max_completion_tokens `
  --json-mode --timeout 120 --repeats 1 --warmup 0 `
  --dataset-kind human_test_set --output-dir output/evaluation/chat_baseline_001
```

在本地受控环境设置`EMR_LLM_API_KEY`，不在命令参数中写入密钥。工具不自动读取`.env`；无认证的本地服务明确使用`--no-auth`，并把部署位置设为`--deployment local`。本轮没有读取真实密钥或调用上述占位服务。正式外部测试应使用已允许发送至该服务的数据；本工具不会删除原文中的身份信息。

`--max-output-tokens`必须明确填写，示例4096不保证足够覆盖实际长病历输出；先用训练/验证部分确认预算及服务上下文范围。输出因预算结束时保留失败。`--token-limit-field`可选`max_tokens`或`max_completion_tokens`，按实际服务契约冻结；不是所有兼容服务都支持同一参数。`--json-mode`仅在服务支持时启用，JSON模式不等于内容正确。可显式设置`--temperature`，未设置时沿用服务默认并记录请求配置，不宣称结果确定。

请求设置固定单候选、非流式，等待完整结果再计时；不以首token响应替代任务完成耗时。服务若只支持Responses等其他协议，需另行适配，当前入口不能宣称通用兼容全部模型。

## 输出、失败与计时口径

| 输出 | 内容 |
|---|---|
| `run_receipt.json` | 输入/XML/规范/提示与工具校验值、模型声明、参数、部署类型及运行状态 |
| `prompt.txt` | 本次冻结系统提示及规范，不含任务原文 |
| `predictions.json` | 首轮正式实体预测，含原文；失败任务保留`status=failed` |
| `structured_predictions.json` | 首轮实体与关系结果，用于内部追溯；不等于关系评分 |
| `benchmark.json` | 正式/预热逐次耗时、字符数、全部失败、成功P50/P95、观察吞吐、服务版本及可得的usage |

复用[专用模型测速客户端](model-service-benchmark.md)相同的单条串行计时循环。耗时含提示请求构造、网络、服务执行、完整响应解析、实体及关系转换和实体校验；任务文件读写、冻结提示准备及预热不计入正式阶段。观察吞吐以整段循环墙钟计，失败占用时间保留；当前不是并发承载测试。

首轮失败不会被后续成功替换。无实体须明确输出两个空数组，空文本、空白或不完整响应不会自动当成功负例。输出达到token上限、拒绝、工具调用、多候选、非法JSON/代码围栏、未知标签、偏移或原文跨度错误、无效关系端点均记失败。不去掉围栏、不搜索原文修补跨度、不请另一个模型改答案后再作为原始输出。

重复预测实体若有不同合法ID会原样保留，评分器计额外误报；重复ID使关系指向含糊，整次输出无效。关系方向统一要求`right`，源/目标按明确格式输出；不猜测反向关系。页面Choices不加入大模型任务。

API返回时记录`prompt_tokens`、`completion_tokens`、`total_tokens`及可得的缓存/推理token；缺失记为`null`，不是0。提示token包含规范、标签和原文，不能当作原文独立token长度。`finish_reason=stop`及usage均不证明服务完整处理全文，截断覆盖仍标未知并结合人工参考核查。

原始HTTP正文与异常正文不写日志；只保留错误类别与状态。预测文件含原文，应留在受控结果目录。工具拒绝覆盖或自动续跑既有目录。真实运行中断时状态记录为`benchmark_interrupted`/`benchmark_failed`；该文件不能证明进程仍在运行，且未完成调用不会自动恢复，需结合进程状态检查，避免无意重复付费请求。完成的预测/测速文件在整轮结束后写出。

## 接入评价和分布图

把`predictions.json`交给[实体评分器](model-evaluation.md)作为独立对照。例如只评价该大模型时，可使用`--before`参数并在实验记录中注明其是通用大模型对照，不把它冒充专用模型的“训练前权重”。现有评分器的一至三路命名为`regex/before/after`，正式汇报中通用大模型与专用模型训练前后是不同实验维度，分别保存结果，不混用名称或省略正则基线。

制图直接读取两侧`benchmark.json`：

```powershell
python scripts/compare_model_benchmarks.py `
  --ours output/evaluation/after_run_001/benchmark.json `
  --baseline output/evaluation/chat_baseline_001/benchmark.json `
  --context output/evaluation/speed_comparison_context.json `
  --plot --log-y --output-dir output/evaluation/speed_comparison_001
```

两侧须读取同一个任务文件，并使用相同单并发、重复次数及正式计时口径；否则图形工具保留统计但不计算比值。成功样本集合不同也不会计算比值。`context.json`应明确同一NER+RE抽取范围和部署位置；外部API阶段不填同本地算力容量。最终本地同资源比较仍需要实际硬件记录、质量及全文覆盖、持续负载与医院工作量。

## 当前验证证据

[测试代码](../tests/test_benchmark_chat_model.py)覆盖严格实体/关系转换、原文与非BMP偏移、重复预测、截断/拒绝/工具调用、多候选、原始文本传递与人工答案隔离、首次失败保留、usage未知值、离线预检及覆盖保护。临时本机HTTP接口实际验证Bearer请求、重定向拒绝、HTTP错误、非法JSON及完整CLI到输出再到制图摘要的衔接，响应均为虚构。

协议参考：[官方Chat Completions请求参数](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)和[完成原因与响应字段](https://developers.openai.com/api/reference/resources/chat)。这些资料支持接口定义，不证明本项目适配已通过某个真实供应商或模型的验证。
