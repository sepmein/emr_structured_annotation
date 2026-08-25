# EMR 结构化标注框架

### —— 基于 Label Studio 的电子病历信息抽取数据集构建体系

---

## 一、项目定位

本项目旨在构建一套面向电子病历（EMR）的**结构化标注与数据集构建框架**，用于支持临床文本信息抽取（Information Extraction）、实体识别（NER）、关系抽取、风险评估建模及知识图谱构建等应用场景。

项目核心目标不是单一标注配置文件，而是建立一个**可扩展、可复用、可规范化的临床标注基础库**。

---

## 二、建设目标

本仓库围绕以下四个核心目标展开：

1. **规范化标注体系构建**
   建立统一的标签体系、定义边界规则和标注逻辑，降低标注歧义。

2. **复杂临床文本问题处理机制**
   构建适用于 EMR 的否定识别、同义归一、时序表达识别等规则框架。

3. **数据资产标准化输出**
   支持标准化 JSON、结构化数据库格式输出，便于模型训练与知识建模。

4. **可扩展的多病种框架设计**
   支持在现有结构上快速扩展至其他疾病或综合症监测场景。

---

## 三、适用场景

本框架适用于以下业务或科研场景：

* 临床自然语言处理（Clinical NLP）
* 监测预警数据集构建
* 病种风险评估模型训练

---

## 四、技术体系结构

整体流程设计如下：

```
原始 EMR 文本
      ↓
脱敏与结构整理
      ↓
Label Studio 标注
      ↓
结构化数据导出
```

## EMR 数据拼合

以 `data/temp_202608_emr_activity_info.csv` 为主表，按
`patient_id + serial_number` 生成 Label Studio 可导入的就诊级 JSON 任务数组：

```powershell
uv run python scripts/merge_emr_data.py
```

脚本仅使用 Python 标准库；如果本机 `uv` 不可用，也可直接运行
`python scripts/merge_emr_data.py`。

默认生成：

- `output/emr_merged_202608.json`：Label Studio 可直接导入的 JSON 任务数组，顶层格式为 `[{"data": {...}}, ...]`。`data.text` 按配置顺序拼合所有指定的非空临床文本，字段标题使用 `【主诉】`、`【现病史】` 等简短中文；已出现过的完全相同内容不会重复拼入。`data.patient_id`、`data.patient_name` 和 `data.text` 分别对应标注配置中的 `$patient_id`、`$patient_name` 和 `$text`。`data.age` 使用身份证出生日期和就诊日期计算，脱敏身份证只能按出生年份估算，此时 `data.age_is_approximate` 为 `true`。`data.age_group` 按年龄小于18岁归为 `儿童组`，否则归为 `成人组`。同一就诊下的活动、文书、医嘱、检验和检查仍以数组保留在 `data` 中。检验、检查和医嘱父记录按 `id` 分组为 `records[]`，其明细放在 `items[]`。
- `output/emr_merged_202608_children.json`：仅包含 `age < 18` 的儿童组。
- `output/emr_merged_202608_adults.json`：仅包含 `age >= 18` 的成人组。
- `output/emr_merge_report_202608.json`：每张表的行数、匹配数、未匹配数及匹配率。

任务文件示例：

```json
[
  {"data":{"patient_id":"p1","patient_name":"患者甲","serial_number":"s1","text":"【主诉】\n发热三天","age":35,"age_group":"成人组","activity_info":[],"patient_info":[],"encounter_data":{}}}
]
```

可通过 `--data-dir`、`--main-file`、`--output` 和 `--report` 指定其他批次或输出位置。运行测试：

```powershell
uv run python -m unittest discover -s tests -v
```

---

## 五、标注体系设计原则

### 1. 结构优先

强调可机器处理结构，而非单纯人工阅读理解。

### 2. 规则可解释

每一个标签类别均需有清晰定义与边界说明。

### 3. 语义分层

支持：

* 实体层
* 属性层
* 关系层
* 时序层

### 4. 可扩展性

标签体系设计采用模块化结构，便于未来扩展至多病种、多系统。
