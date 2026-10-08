# GLiNER 旧技术路线退役记录

日期：2026-10-08。根据负责人确认的技术路线调整，移除 GLiNER 专属代码，保留 ModernBERT、数据整理与标注体系。

## 删除范围

| 原路径 | 用途 |
|---|---|
| `ml_backend/model.py` | GLiNER2 加载、旧文本拼接、实体与关系预标注 |
| `ml_backend/prompts.py` | GLiNER 固定标签提示和关系描述 |
| `ml_backend/_wsgi.py` | GLiNER Flask 服务入口 |
| `ml_backend/test_predict.py` | GLiNER 实际模型调试 |
| `ml_backend/__init__.py` | 旧后端包入口 |
| `main.py` | GLiNER2 英文样例冒烟脚本 |

移除根 `pyproject.toml` 中的 `gliner`、`gliner2`，并由 `uv lock` 更新锁文件。仅随其失去引用的 `coloredlogs`、`flatbuffers`、`humanfriendly`、`onnxruntime`、`protobuf`、`pyreadline3` 从根依赖图移除；剩余包版本保持原锁定版本。

## 保留与安全依据

- 删除前上述源码、`pyproject.toml`、`uv.lock` 均无未提交改动，源码存在于下述 Git 提交中。此次不提交、不改写 Git 历史。
- 全仓引用搜索与保留代码的 AST 导入检查未发现 ModernBERT、数据整理或试标工具运行时导入 GLiNER 旧包。历史提案中的路径是记录当时问题的文本，不是代码依赖。
- ModernBERT、XML、数据整理、原有测试、试标历史、技能源码和 notebook 共 184 个原有文件逐文件核对 SHA-256，保持原样。新增的退役回归测试单独保留。
- `Untitled.ipynb` 使用 Transformers，不属于 GLiNER。根环境显式保留原来间接安装的 Torch、Transformers，并保留与 ModernBERT 共用的 SentencePiece 依赖；没有删除 ModernBERT 的独立 `requirements.txt` 或本地 ML 运行时。
- 没有执行 `uv sync`、环境卸载、模型缓存清理、数据或模型权重删除。无代码的本地 `gliner_backend/` 残留目录也未递归删除。
- 历史指南、裁决结果、Schema 提案和历史审查段落保持原样；其中关于补齐 GLiNER 的建议是历史背景，不再是当前路线待办。ModernBERT 设计文档中的旧 `ml_backend/` 路径也不构成运行依赖，当前入口以根 README 为准。
- 根 README、AGENTS.md 和 CLAUDE.md 已同步到 ModernBERT 路线，避免继续调用已删除入口。

## 验证与边界

```sh
python -m unittest discover -s tests -v
uv lock --check --offline --cache-dir /tmp/emr-gliner-uv-cache
python documentation/annotation/scripts/check-document-links.py
```

11 项离线测试通过，包括新增的保留代码导入与依赖退役检查；锁文件一致性、文档链接及保留文件哈希检查通过。`tests/test_no_legacy_gliner.py` 不加载模型、不访问实际服务。

本次核查覆盖仓库代码与依赖，不确认外部 Label Studio 是否仍配置旧 GLiNER 服务地址；未启动、停止或修改已部署的服务，也未验证 ModernBERT 真实推理、训练。旧 GLiNER 服务的源码重启入口已删除，需要恢复旧路线时按下述方式恢复。

## 恢复

删除前基线提交：`7c925cb95d2a37a22711a0acb2330749afc5a2ec`。如需恢复该路线，在仓库根目录执行以下命令；它会还原列出的旧源码与依赖文件，执行前确认这些路径没有后续要保留的改动：

```sh
git restore --source=7c925cb95d2a37a22711a0acb2330749afc5a2ec -- ml_backend main.py pyproject.toml uv.lock
```

此命令不安装依赖或恢复旧 README；旧模型权重与缓存未被本次删除。当前 ModernBERT 运行方式参见 [README](../README.md)。
