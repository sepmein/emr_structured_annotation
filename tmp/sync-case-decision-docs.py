from pathlib import Path
root=Path(__file__).resolve().parents[1]
def change(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old))
    return text.replace(old,new)
guide=root/'annotation_agent_workflow/guides/annotation_guide_v2.1.1.md'
text=guide.read_text(encoding='utf-8')
edits=[
('| 病例级判断 | 外部四分类表单及最小证据 | 主要病例输出 |','| 病例级判断 | 页面顶部“病例四分类”单选控件；最小证据另在外部表单保存 | 主要病例输出 |'),
('只在病例表单中选择“目标病例信号、待专业复核、非目标、信息不足”之一。','在左侧标注栏顶部的“病例四分类”中选择“目标病例信号、待专业复核、非目标、信息不足”之一。该控件必选且没有默认值，作用于整份病例，不随实体选区切换。'),
('| 病例四分类、数据问题 | 无统一控件 | 外部表单记录 |','| 病例四分类 | 页面顶部必填单选控件 | 按第3章选一个结论，无需框选实体 |\n| 原文证据、任务状态、数据问题 | 无相应控件 | 仍在外部表单记录 |'),
('49个实体标签、9个实体属性字段、5种关系。属性选项不是实体标签。','49个实体标签、9个实体属性字段、5种关系，以及1个病例级单选控件。病例四分类和实体属性选项均不是实体标签。'),
('- 按统一配置核对49个实体标签、9个属性字段及5种关系，同步框选规则、案例操作、关系方向、提交检查和质量评价；不再按患者年龄区分页面操作。','- 按统一配置核对49个实体标签、9个属性字段及5种关系，同步框选规则、案例操作、关系方向、提交检查和质量评价；不再按患者年龄区分页面操作。\n- 增加病例级必填单选控件`case_decision`，提供四种病例结论，不设默认值、不绑定实体选区；原文证据、任务状态和数据问题仍在外部表单记录。'),
('| 病例判断与数据问题 | 外部表单 | 四分类、0—3条必要证据及问题 |','| 病例判断 | 页面顶部“病例四分类” | 必选一个结论，无默认值；字段名`case_decision` |\n| 病例证据与数据问题 | 外部表单 | 保存0—3条必要证据及任务状态、数据问题 |')]
for old,new in edits:text=change(text,old,new)
guide.write_bytes(text.replace('\n','\r\n').encode('utf-8'))
dictionary=root/'annotation_agent_workflow/guides/label_dictionary_v2.1.1.md'
text=dictionary.read_text(encoding='utf-8')
edits=[
('- F：外部病例表单字段，直接在表单中填写，无需在原文中框选或连接关系。','- CF：页面顶部的病例级单选控件，每份病例选一个结论，无需框选实体或连接关系。\n- F：外部病例表单字段，直接在表单中填写，无需在原文中框选或连接关系。'),
('包含49个实体标签、9个属性字段和5种关系。','包含49个实体标签、9个实体属性字段、5种关系，以及1个病例级单选控件。'),
('2. 按主指南选择病例结论。','2. 按主指南在页面顶部的“病例四分类”中选择一个结论。'),
('四类的具体判定顺序见主指南第3章。','四类的具体判定顺序见主指南第3章。\n\n在左侧标注栏顶部的“病例四分类”中选择，控件名为`case_decision`。该控件必选、单选、无默认值，作用于整份病例，不绑定实体选区。结论随Label Studio标注结果导出，原文证据仍另存外部表单。发现身份信息时停止该例，不选择结论、不保存证据、不提交。'),
('本次核对XML能否解析、各控件名称与文本目标是否对应，以及字典是否覆盖49个实体标签、9个属性字段的全部选项和5种关系。属性必填要求及适用实体按XML逐项核对。','本次核对XML能否解析、各控件名称与文本目标是否对应，以及字典是否覆盖49个实体标签、9个实体属性字段的全部选项、5种关系和1个病例级四分类控件。四分类控件核对单选、必填、无默认值及整份病例作用范围；实体属性按XML核对必填要求和适用实体。'),
('当前XML未提供病例四分类、任务状态或数据问题控件，这些内容仍在外部病例表单记录。','当前XML已提供病例四分类控件；原文证据、任务状态和数据问题仍在外部病例表单记录。')]
for old,new in edits:text=change(text,old,new)
start=text.index('## 2. 病例级判断标签');end=text.index('## 3. 核心肺部证据')
cards=text[start:end]
assert cards.count('病例表单必填')==4
cards=cards.replace('病例表单必填','CF：页面顶部病例四分类必选')
text=text[:start]+cards+text[end:]
dictionary.write_bytes(text.replace('\n','\r\n').encode('utf-8'))

builder=root/'documentation/layout-preview/build-annotation-guide.cjs'
text=builder.read_text(encoding='utf-8')
text=change(text,'49个实体标签、9个属性字段、5种关系。两类任务使用同一套页面操作','49个实体标签、9个实体属性字段、5种关系及1个病例级单选控件。两类任务使用同一套页面操作')
builder.write_text(text,encoding='utf-8')

preview=root/'documentation/layout-preview/annotation-manual-preview.html'
text=preview.read_text(encoding='utf-8')
text=change(text,'病例结论、原文证据和数据问题已在外部表单记录。','页面顶部已选择一个病例结论；原文证据和数据问题已在外部表单记录。')
# This quick reference has no case cards, so add a clear link to the guide's classification control instructions.
needle='<li>在外部病例表单中保存证据，记录问题。</li>'
text=change(text,needle,'<li>在页面顶部“病例四分类”中选一个结论，不设默认值。</li>'+needle)
preview.write_text(text,encoding='utf-8')
dictionary_html=root/'documentation/layout-preview/annotation-manual-design-b.html'
text=dictionary_html.read_text(encoding='utf-8')
text=change(text,'请与主指南共同使用。','页面顶部“病例四分类”必选一个结论；证据和数据问题另存外部表单。')
dictionary_html.write_text(text,encoding='utf-8')
print('Guide, dictionary and HTML instructions synchronized with the case-level control.')
