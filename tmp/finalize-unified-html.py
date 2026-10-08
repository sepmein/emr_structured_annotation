from pathlib import Path
root=Path(__file__).resolve().parents[1]
builder=root/'documentation/layout-preview/build-annotation-guide.cjs'
text=builder.read_text(encoding='utf-8')
assert text.count("'儿童关系方向示例'")==1
text=text.replace("'儿童关系方向示例'", "'关系方向示例'")
old='<aside class="version-note" role="note"><strong>阅读前请核对配置版本</strong>本页保留主指南原稿。原稿中仍有成人、儿童分开配置及旧字典状态的说明，尚未与当前统一配置同步。按钮、属性和关系范围请同时核对<a href="../../documentation/layout-preview/annotation-manual-design-b.html">新版标签字典</a>与<a href="../../label_studio/pneumonia_config.xml">当前 XML</a>；病例判断仍按主指南第3—4章执行。</aside>'
new='<aside class="version-note unified" role="note"><strong>成人与儿童共用统一配置</strong>本指南与<a href="../../documentation/layout-preview/annotation-manual-design-b.html">标签字典</a>均已按<a href="../../label_studio/pneumonia_config.xml">pneumonia_config.xml</a>同步：49个实体标签、9个属性字段、5种关系。两类任务使用同一套页面操作，病例判断按第3—4章执行；儿童特有体征仍按其医学定义和纳入范围标注。</aside>'
assert text.count(old)==1
text=text.replace(old,new)
text=text.replace('@media(min-width:1700px){','.version-note.unified{background:#edf5f1;border-color:#73a08a;color:#496e5f}\n@media(min-width:1700px){')
builder.write_text(text,encoding='utf-8')
dictionary=root/'documentation/layout-preview/annotation-manual-design-b.html'
html=dictionary.read_text(encoding='utf-8')
old='教学展示，请与主指南共同使用。'
new='<a href="../../annotation_agent_workflow/guides/annotation_guide_v2.1.1.html" style="color:#d7e8e0">查看主指南 · 统一配置 ↗</a><br>请与主指南共同使用。'
assert html.count(old)==1
dictionary.write_text(html.replace(old,new),encoding='utf-8')
print('HTML builder status, flowchart caption, and dictionary cross-link synchronized.')
