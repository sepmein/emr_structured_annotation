# 项目阶段汇报 PPT 维护入口

汇报脚本为 [project_progress_presentation.mjs](../../emr_annotation/evaluation/project_progress_presentation.mjs)。
脚本读取已复核的汇总快照、人工冻结清单、流程核查回执及检测属性审计，不读取或展示真实病历正文。
Schema 检测记录示例是虚构示意。具体结果、PPT、输入哈希和验证回执均留在对应 output 运行，不放入源码。

## 文件与内容

运行位置采用 `output/<batch_id>/delivery/<run_id>/`；本轮入口是
`output/schema_training_trial_20261011_v001/delivery/run_presentation_001/`。

- `final/`：可交付的 PPTX。
- `storyline.json`：页面逻辑及每页证据来源。
- `presentation_manifest.json`：输入、脚本和 PPTX 哈希，以及实际验证状态。
- `assets/`：封面概念插画与来源、生成提示词。
- `.codex_build/`：私有构建副本、候选文件、最终文件渲染及验证回执，不作为用户交付材料。

采用 16:9、微软雅黑、可编辑文字和原生图表，突出统一规范、人工控制、证据留存和实测结果。
最后两页分别说明不足和下一步。目标尚未实现时必须保持阶段汇报定位，不将最高测试分数改写为验证选中模型的结果。
病例准确率、病例 macro F1、实体严格匹配 F1及属性已知槽位正确比例分开描述。

## 重建方式

使用本地 bundled Node 与 `@oai/artifact-tool`，将脚本复制至该运行的私有构建目录，使其能够解析运行时依赖。
设置 `RUNTIME_NODE_MODULES` 指向已安装的 bundled Node modules。脚本参数如下：

```text
--snapshot <reviewed_snapshot.json>
--freeze <freeze_manifest.json>
--verification <pipeline_verification.json>
--attributes <pathogen_attributes.json>
--workspace <新的绝对运行目录>
--cover <封面概念插画.png>
--skill-dir <presentations 技能绝对目录>
--runtime-python <bundled Python 路径>
--output-name <新的 PPTX 文件名，可省略>
```

已有最终文件不覆盖。修改后使用新的 run_id 或文件名，保留旧版本。
脚本调用技能 finalizer，核查包结构、页面几何、字体声明、原生图表及内嵌数据，然后重新导入最终 PPTX 渲染每一页。
图表单元格为兼容 Excel 保留 15 位有效数字，输入快照及其哈希保持原样。

导出后还需逐页查看最终文件的渲染、核对图表数据与来源，并将实际检查写入运行清单。
结构检查及 Artifact Tool 渲染通过不等同于在原生 PowerPoint 中打开验收，不得将未执行的检查写为完成。
