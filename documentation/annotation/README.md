# 肺炎标注资料

标注指南、标签字典及其 HTML 阅读版是一套配套资料。先读指南确定病例判断和操作流程，再查字典确定实体跨度、属性和关系；HTML 为阅读入口，Markdown 为维护原稿。内容版本及草案状态以各原稿说明为准，文件迁移不改变发布状态。

| 资料 | HTML 阅读版 | Markdown 原稿 |
|---|---|---|
| 标注指南 | [guide.html](guide.html) | [annotation_guide_v2.1.1.md](annotation_guide_v2.1.1.md) |
| 标签字典 | [label-dictionary.html](label-dictionary.html) | [label_dictionary_v2.1.1.md](label_dictionary_v2.1.1.md) |

两份原稿保留原文件名，正文已有更新版本说明，不应只按文件名判断内容版本。标签字典 HTML 保留原设计评审说明。

配套的 Label Studio 页面配置为 [pneumonia_config.xml](../../label_studio/pneumonia_config.xml)。

默认配置保留旧实体控件名，各组内单选，兼容已有标注；病例结论和属性按各自字段单选。新项目可采用[跨分组单选版](../../label_studio/pneumonia_config.global-single.xml)。操作与配置选择见[单选配置与旧项目兼容说明](label-studio-single-selection.md)。

## 目录

```text
annotation/
├── README.md
├── guide.html                         # 标注指南阅读版
├── label-dictionary.html              # 配套标签字典阅读版
├── annotation_guide_v2.1.1.md          # 指南原稿
├── label_dictionary_v2.1.1.md          # 字典原稿
├── scripts/
│   ├── build-annotation-guide.cjs      # 从指南原稿生成 guide.html
│   └── check-document-links.py         # 检查本地链接与静态锚点
└── archive/layout-preview/            # 旧排版样稿及设计截图
```

[历史指南、字典、验证报告和修订记录](../../annotation_agent_workflow/guides)保留在原目录，以便追溯历史试标与裁决依据。

## 维护

修改指南原稿后，从仓库根目录运行：

```sh
node documentation/annotation/scripts/build-annotation-guide.cjs
```

脚本需要已有的 `marked` 包，也支持本机 Codex 自带依赖。标签字典 HTML 当前是独立维护的页面，修改字典原稿后需同步核对页面；上述命令只生成指南。

移动或修改链接后，可运行 `python documentation/annotation/scripts/check-document-links.py` 检查本地文件与静态锚点。字典页面动态生成的章节锚点需在浏览器中检查。
