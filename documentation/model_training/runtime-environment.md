# 本地训练与评估环境

根 `.venv` 用于数据准备、notebook 和离线工具；ModernBERT 使用独立的、符合 [requirements.txt](../../modernbert_ml_backend/requirements.txt) 的环境。两套依赖分别维护，不在根环境安装后端 requirements。

## 当前本机配置（2026-10-10）

根环境此前实际解释器为 Python 3.13.3，但已安装的扩展是 cp312，导致 PyTorch 导入失败。已将 `pyvenv.cfg` 指向已有 Python 3.12.10，并刷新解释器时间戳以使 uv 重新检测；原环境文件已备份，没有替换已有包或终止既有进程。另以 `uv sync --inexact` 补齐锁文件中缺失的 78 项依赖，保留已有包；再次 dry-run 显示无需变更。新启动的根 Python/uv 能加载 Torch 2.10.0+cpu、Transformers 5.0.0；136 个已安装包依赖检查、CPU 反向传播、实际 Jupyter kernel 执行及当前工作区 154 项离线测试通过。既有 Python 进程需在方便时重新启动才能使用修复后的解释器。

GPU 环境位于 `D:/codex_envs/emr_structured_annotation/modernbert_py312_cu126`，仓库中的 `modernbert_ml_backend/.venv` 是指向它的本地 Junction，且被 Git 忽略。下载缓存位于 `D:/codex_cache/emr_structured_annotation`。这样 E 盘继续用于数据、模型权重和运行产物。迁移到其他电脑时需要重新创建环境及 Junction，不能只复制这个链接。

本机为 i7-12700、约 32 GB 内存及 RTX 4070 Ti SUPER（约 16 GB 显存）。后端已安装 Python 3.12.10、Torch 2.7.1+cu126、Transformers 5.5.4、PEFT 0.19.1 和其余固定依赖，94 个包依赖检查及 41 项实际运行检查全部通过。CPU/GPU 反向传播、AdamW 更新、CUDA 混合精度和 GradScaler 均已执行。CUDA wheel 来源见 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)；本次下载按官方索引 SHA256 校验通过。

迁移后的训练入口已在此 GPU 环境用现有虚构 tiny 模型完成 3 轮训练，验证集选择第 2 轮，实际保存和重载后完成同一测试集的前后评估。训练损失由 4.0525 降至 0.1134，保存的 29 个权重张量均为有限值，LoRA B 矩阵有 16384 个非零元素。虚构测试集实体 micro F1 从 0% 到 93.0%，端到端关系 micro F1 从 0% 到 88.2%；只覆盖 5/49 个实体标签，使用重复模板。此运行用于环境及流程验收。训练记录位于 `output/environment_check_20261010_v001/training/run_001/`，环境检查位于同批次 `evaluation/gpu_001/`。

`_wsgi.py --help` 的本地服务命名空间和 SDK 导入通过；未启动服务、调用服务路由或开展 Label Studio UI 测试。

本次审计位置为 `output/environment_check_20261010_v001/evaluation/run_001/`。其中 `environment_receipt.json` 记录修复前后状态、源文件哈希、代码版本、原有未提交改动和解释器备份。历史模型迁移失败回执保持原内容，本次修复另行记录。

## 日常使用

从仓库根目录执行，明确使用后端环境解释器；不要用根目录的 `uv run` 启动模型训练。

```powershell
# 每次使用新的 run_id；检查会拒绝已有输出目录
& .\modernbert_ml_backend\.venv\Scripts\python.exe `
  modernbert_ml_backend/check_environment.py --require-cuda `
  --output-dir output/environment_check_20261010_v001/evaluation/run_002

# 参数说明不导入模型库、不加载权重
& .\modernbert_ml_backend\.venv\Scripts\python.exe `
  modernbert_ml_backend/train_adjudicated.py --help
```

[环境检查入口](../../modernbert_ml_backend/check_environment.py)逐项验证固定依赖版本及实际导入，执行 CPU/GPU 前向、有限梯度反向传播和 AdamW 更新，并测试 CUDA 混合精度及 GradScaler。任一必需检查失败即返回非零状态；完整结果在新目录的 `runtime_check.json`。检查不加载权重、不发送模型请求，不代表训练数据、模型性能或服务已验收。

训练命令参照[裁决后训练与前后评估](adjudicated-model-training.md)，将 `python` 替换为上面的后端解释器。基础权重仍放在 `pretrained_models/<base_model>/`，新产物使用 `output/<batch_id>/training/<run_id>/`；三份冻结集合与配置快照必须对应同一个最终参考。根环境继续通过 `uv run` 执行数据工具。

## 恢复独立环境

示例使用当前本机 D 盘路径。先创建环境、安装 CUDA Torch，再安装其余固定依赖；第二步保留满足固定版本的 CUDA Torch。每步完成后检查实际版本，最后运行 `check_environment.py --require-cuda`。

```powershell
$env:UV_CACHE_DIR = 'D:/codex_cache/emr_structured_annotation'
$env:UV_HTTP_TIMEOUT = '600'
uv venv D:/codex_envs/emr_structured_annotation/modernbert_py312_cu126 --python 3.12
uv pip install --python D:/codex_envs/emr_structured_annotation/modernbert_py312_cu126/Scripts/python.exe `
  'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python D:/codex_envs/emr_structured_annotation/modernbert_py312_cu126/Scripts/python.exe `
  -r modernbert_ml_backend/requirements.txt
# 在本地链接不存在时创建；不要覆盖已有环境或链接
New-Item -ItemType Junction -Path modernbert_ml_backend/.venv `
  -Target D:/codex_envs/emr_structured_annotation/modernbert_py312_cu126
```

## 正式模型训练前提

2026-10-10 环境验收时只发现已有的虚构 tiny ModernBERT smoke 模型；该记录不代表正式中文权重已训练。2026-10-11 已另行下载官方中文BERT到`pretrained_models/bert_base_chinese/`，并以独立跨度多任务入口完成原始双标试验训练及重载，见[本轮入口与验证边界](schema-multitask-trial.md)。最终人工裁决冻结数据及合格关系标注仍需准备；已有smoke数据只用于流程验证，不能从tiny模型推断正式模型的容量和医学效果。

网络同时输出 BIO 实体和实体对关系，不包含 9 个实体属性及病例级选择的输出头。真实模型性能、医学裁决质量、Label Studio UI、服务多项目状态隔离仍需分别验收。
