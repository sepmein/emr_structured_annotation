# 最终裁决结果的 ModernBERT 训练与前后评估

入口：[train_adjudicated.py](../../modernbert_ml_backend/train_adjudicated.py)。复用现有 ModernBERT 共享编码器、BIO 实体头、实体对关系头、LoRA、联合损失及动态填充，增加完整实验流程：冻结数据 → 核查 → 训练前测试 → 训练 → 验证集选最佳模型 → 重载 → 同一测试集训练后测试。

2026-10-10 源码迁移：共享网络在 [modeling/network.py](../../modernbert_ml_backend/modeling/network.py)，数据集/训练/保存函数在 [training/trainer.py](../../modernbert_ml_backend/training/trainer.py)，训练实现不依赖 Label Studio 服务。旧 `train_ner_re.py` 保留别名，顶层 `config` 与脚本命名空间不变。迁移验证及模型环境尝试的失败范围见[模型迁移快照](../model-migration.json)；历史虚构 smoke 记录保持，本次迁移后的实际模型运行仍需在指定独立环境验证。

当前目标为 XML 中 49 个实体标签和 5 种关系。9 个实体属性字段和病例级选择保存在最终参考中，没有对应模型输出头，不能将本流程结果作为它们的训练效果。

## 1. 准备唯一的最终参考

输入是标准化的 `reference.json`，不是仍含两个人答案的原始 Label Studio 导出，也不是差异表或尚未提交的裁决草稿。每条病例需已完成全部实体与关系复核，包括确认没有目标实体/关系的负例。

以下全部为虚构示例，用于说明格式：

```json
[
  {
    "task_id": "batch01-project230-task001",
    "patient_id": "internal-patient-001",
    "text": "体温38℃",
    "entities": [
      {"id": "e1", "start": 0, "end": 2, "label": "体温"},
      {"id": "e2", "start": 2, "end": 5, "label": "数值"}
    ],
    "relations": [
      {"from_id": "e1", "to_id": "e2", "type": "测量", "direction": "right"}
    ],
    "attributes": [],
    "case_choices": {}
  }
]
```

`task_id` 在汇总所有批次、项目后必须唯一，`patient_id` 必须能跨批次识别同一患者，不能用项目内任务 ID 代替患者 ID。`text` 保留原文，偏移使用 Python 字符位置 `[start,end)`；非 BMP 字符需先确认平台偏移单位。负例显式写 `entities: []` 与 `relations: []`，缺字段不视作完成复核。

截至 2026-10-09，引用的双标核查聊天已指出原始导出的关系只有端点/方向、缺少类型；最终裁决必须补齐 `type`，不能猜测或默认为“无关系”。单层 BIO 不支持重叠实体，单关系分类头不支持同一有向实体对多类型；入口对训练/验证中的这些情况报错，需要先明确医学和监督策略。

如裁决结果存为 Label Studio 新增的最终 annotation，使用[参考准备工具](../training_data_preparation/label-studio-evaluation-preparation.md)明确选取最终 annotation；不要默认取 `annotations[0]`。直接生成标准化参考时，另外保留裁决人、版本、理由和原结果 ID 等来源记录。程序验证结构与文件一致性，不认证医学裁决。

XML 必须是这些参考使用的配置快照。已有项目使用保留旧控件名的配置；新项目可使用跨组单选配置。不得为适应当前 UI 重写历史标注或已冻结的划分清单。

## 2. 汇总后一次冻结患者分组

```powershell
python scripts/split_evaluation_reference.py `
  --reference output/adjudicated/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --ratios 0.7 0.15 0.15 --seed 42 `
  --output-dir output/adjudicated/splits_v1
```

比例是起点，分组后的实际条数和标签支持以清单为准。工具把同一患者和精确相同全文组成不可拆分的组；训练、验证、测试互不交叉。少量数据无法保证每类都有测试支持，应检查 `split_manifest.json` 的 coverage warnings；验证必须有阳性训练目标。近重复还需人工排查。

后续批次纳入训练时，保留既定测试患者及其近重复隔离，不把曾看过的测试结果反复用于调参。入口只接受一个锁定参考及其对应三份集合，不自动重划分、扩样或拼接历史批次。

## 3. 无模型核查

```powershell
python modernbert_ml_backend/train_adjudicated.py `
  --splits-dir output/adjudicated/splits_v1 `
  --reference output/adjudicated/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/adjudicated/preflight_v1 --check-only
```

核查 SHA256、清单归属、原文、全部参考字段、患者/全文隔离、实体与关系结构，并写 `experiment_receipt.json`。不导入 Torch/Transformers，不加载 tokenizer/权重，也不生成性能数字。真实运行还会用本次 tokenizer 核查全部三份集合：全文必须在最大长度内，实体边界必须精确可表示，实体不能共占 token。问题明确失败，不静默截断或修改答案。超长病例需采用基础模型支持的更大长度，或另行设计固定窗口与跨窗口关系协议。

## 4. 训练并完成同口径前后评估

在满足 [ModernBERT requirements](../../modernbert_ml_backend/requirements.txt) 的独立 Python 环境中，以脚本方式运行。根目录环境不替代此依赖环境。本地权重需放在 `pretrained_models/<base-model>/`。

```powershell
python modernbert_ml_backend/train_adjudicated.py `
  --splits-dir output/adjudicated/splits_v1 `
  --reference output/adjudicated/reference.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/adjudicated/run_v1 `
  --base-model chinese-modernbert-large-wwm `
  --epochs 20 --batch-size 4 --max-length 1024 --seed 42 `
  --learning-rate 0.00005 --classifier-lr 0.0005 --patience 5 `
  --ner-threshold 0.5 --re-threshold 0.5 --re-max-distance 50
```

默认“训练前”为本地预训练编码器加随机初始化的任务分类头，是初始化诊断。训练从被评估的同一模型对象继续，不另起一套随机分类头。如果要评估已有任务模型，并在它的基础上继续训练，加 `--initial-checkpoint <已有 best_model 的路径>`；不恢复历史 optimizer、epoch 或历史最佳分。

已有 checkpoint 必须包含 `label_mappings.json`，其 `schema_id`、`ner_label2id`、`re_label2id` 和 `base_model_name` 与本次冻结配置一致，并包含完整 LoRA adapter + 分类头，或唯一的全参数权重。旧产物缺少映射时，需要根据该模型当时的配置和训练记录审计后补充；不能用当前 XML 猜测标签顺序。LoRA 继续训练加载使用 `is_trainable=True`，按 [PEFT 官方说明](https://huggingface.co/docs/peft/main/tutorial/peft_model_config)显式开启 adapter 训练。全参数旧模型作为初始化时，增加新 LoRA，再训练分类头和 adapter。

缺少 checkpoint 或映射不回退成随机模型。已有输出目录报错，避免覆盖/意外续训；每次使用新目录。该入口独立于原来的服务 `fit()` 和 `train_frozen.py`，不改它们的默认行为。

## 5. 模型选择与指标

训练前后固定同一测试原文、标签映射、阈值、最大长度和关系距离限制。预测完全不读取人工实体端点。每轮仅预测验证集，选取**严格实体 micro F1 和端到端关系 micro F1 的有支持目标均值**，按最小提升阈值和 patience 早停；某目标没有验证阳性时不伪造该目标分数，也不参与均值。首轮即使得分为 0 也保存。

训练后释放最后一轮模型和 optimizer，实际重载所选 `best_model/` 后才生成测试预测。比较因此针对保存的最佳产物。测试分数不用于选轮次或动态调阈值。

实体匹配要求字符起止与实体标签一致。关系匹配要求关系类型、方向，以及两个端点的字符起止和实体标签全部一致；实体漏检会成为关系漏检。全部未标注有向实体对视为负例，因此最终关系复核必须完整。正负关系的随机采样只用于现有训练损失，不用于最终评分。

报告逐类支持数、TP/FP/FN、P/R/F1、总体 micro F1、仅有参考支持类别的 macro F1，以及训练前后总体 micro F1 的变化百分点。重复预测按一对一匹配计额外 FP；没有测试参考的标签显示 `null`/“—”，误报仍计入总体。离线独立评分时，缺失或失败任务保留分母；自动实验中预测失败则整次实验标失败，不伪造成功空预测。

保留现有模型的实体解码、同类型实体对排除和字符距离过滤，另外屏蔽 tokenizer 的零跨度特殊 token，避免生成无效字符实体。两次评估使用相同实现；若与现有服务比较，需同时核对特殊 token 的处理。这些规则会限制部分合法关系，逐标签结果中保留因此产生的漏检。

## 6. 产物与复核

| 产物 | 内容 |
|---|---|
| `experiment_receipt.json` | 输入/代码/权重校验值、映射来源、参数、运行版本、状态、所选轮次、产物校验值 |
| `token_audit.json` | 各集合长度和实体可表示性核查 |
| `before_predictions.json`、`before_metrics.json` | 训练前测试集原始预测与严格评分 |
| `validation_history.jsonl`、`training.log` | 训练损失、每轮两项严格验证指标、选择分和早停日志 |
| `best_model/` | adapter、分类头、checkpoint、完整标签映射与基础模型名 |
| `after_predictions.json` | 重载最佳模型后的测试集预测 |
| `comparison.json`、`comparison.md` | 49 类实体与 5 类关系的前后对比 |

预测保留原文，只放在受控、Git 忽略的 `output/` 下。失败会保留回执、失败类型和已有日志；只有重载及前后评分完成后才写 completed 状态。比较结果用于评估，模型不会自动替换现有服务。

已取得两个模型的预测后，也可离线重新评分：

```powershell
python scripts/evaluate_ner_re.py `
  --reference output/adjudicated/splits_v1/test.json `
  --before output/adjudicated/run_v1/before_predictions.json `
  --after output/adjudicated/run_v1/after_predictions.json `
  --label-config label_studio/pneumonia_config.xml `
  --output-dir output/adjudicated/rescore_v1
```

离线测试使用虚构数据与模拟模型，不证明真实医学质量、GPU 训练、真实权重重载或效果改善。真实模型效果和服务速度需拿到最终参考与本地权重后运行；本入口不包含速度测量。

## 7. 实际反向传播流程测试

新增[虚构数据与小模型构建工具](../../scripts/build_training_smoke_fixture.py)，生成144条八种模板的虚构记录及随机初始化的2层 ModernBERT（hidden_size=64），使用字符 tokenizer。它用于验证真实反向传播、LoRA优化、保存和重载；没有使用预训练中文权重。默认 XML 为跨组单选版，包含全部49实体和5关系；模板实际只覆盖5个实体标签及5种关系。

在独立的锁定依赖环境中构建，再调用同一训练入口。下面使用新的目录与模型名，避免覆盖已完成的实验：

```powershell
python scripts/build_training_smoke_fixture.py `
  --output-dir output/adjudicated_smoke_002/fixture `
  --base-model-name synthetic-modernbert-smoke-002 --tasks 144 --seed 42

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:MODEL_DIR = "output/adjudicated_smoke_002/backend_cache"
python modernbert_ml_backend/train_adjudicated.py `
  --splits-dir output/adjudicated_smoke_002/fixture/splits `
  --reference output/adjudicated_smoke_002/fixture/reference.json `
  --label-config output/adjudicated_smoke_002/fixture/label_config.xml `
  --output-dir output/adjudicated_smoke_002/run_001 `
  --base-model synthetic-modernbert-smoke-002 `
  --epochs 40 --batch-size 8 --max-length 128 --seed 42 `
  --learning-rate 0.001 --classifier-lr 0.003 --patience 8
```

小模型的学习率是流程测试参数，不应直接用于大规模预训练编码器。运行回执将此类数据标记为 `synthetic_smoke`，训练前基线标记为 `random_tiny_encoder_and_random_task_heads`。即使模板测试得分很高，也不能解释为真实病历泛化能力；44个未覆盖标签仍记为无支持。

本次实际CPU测试已经完成，结果保存在受控的 `output/adjudicated_smoke_20261009_001/run_001/summary.md`。权重、原始预测、输入快照和实际执行时的代码快照一起保留；这些实验产物由Git忽略。目录日期表示实验开始日期。
