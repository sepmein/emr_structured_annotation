from pathlib import Path
import re,json
root=Path(__file__).resolve().parents[1]
def change(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new,1)
guide=root/'annotation_agent_workflow/guides/annotation_guide_v2.1.1.md'
text=guide.read_text(encoding='utf-8')
old='发现身份信息时停止，不选结论、不复制原文。\n\n### 2.3'
new='''发现身份信息时停止，不选结论、不复制原文。

#### 在页面上怎样选择与提交

1. 阅读全文并按第3—4章判断后，在左侧标注栏顶部选择一个结论。四个选项互斥，不需要先框选原文或选中实体。
2. 完成实体、属性和关系标注后，回到“病例四分类”核对所选结论与证据是否一致。需要更正时，改选另一个选项。
3. 提交前确认已选择一个结论。该控件没有默认值，未选择时会提示补选；不能为绕过必填检查而随意选择“非目标”或“信息不足”。
4. 将必要原文证据、任务状态和数据问题另存外部表单，并与该任务对应。四分类控件只保存结论，不保存证据片段。

病例结论在Label Studio中记录为`case_decision`，适用于整份病例。它不是原文实体，不填写存在状态、时间属性等实体属性，也不与实体连接关系。外部流程若另有病例结论字段，须与页面所选结论一致。

### 2.3'''
text=change(text,old,new)
text=change(text,'3. 按图1选择一个病例结论','3. 在病例四分类中选择一个结论<br/>判定顺序见图1')
text=change(text,'7. 按结论要求保存0—3条<br/>最短必要证据','7. 在外部表单中按结论要求<br/>保存0—3条最短必要证据')
text=change(text,'- 病例级只选了一个结论；','- “病例四分类”已选择一个结论，且与原文支持证据一致；')
text=change(text,'正式开始前先验证统一配置页面：适用实体选中后面板显示','正式开始前先验证统一配置页面：病例四分类只能单选、无默认值、未选择时阻止提交，切换实体后仍保留该病例结论；适用实体选中后面板显示')
guide.write_bytes(text.replace('\n','\r\n').encode('utf-8'))

dictionary=root/'annotation_agent_workflow/guides/label_dictionary_v2.1.1.md'
text=dictionary.read_text(encoding='utf-8')
old='发现身份信息时停止该例，不选择结论、不保存证据、不提交。\n\n### 2.1'
new='''发现身份信息时停止该例，不选择结论、不保存证据、不提交。

### 四分类的操作与记录位置

| 要记录的内容 | 在哪里操作 | 提交前核对 |
|---|---|---|
| 病例结论 | 左侧标注栏顶部“病例四分类” | 选一个结论；需要更正时改选其他选项；不能以空白提交 |
| 支持结论的原文证据 | 外部病例表单 | 目标病例信号、待专业复核、非目标各保存1—3条；信息不足保存0—3条 |
| 原文实体、适用属性与关系 | 正文框选、所选实体属性面板及关系工具 | 不将病例结论当成实体或实体属性，不为结论画关系 |
| 任务状态与数据问题 | 外部病例表单 | 与该任务对应；发现身份信息时停止，不能用任一四分类选项代替隐私处置 |

切换实体后，病例结论仍属于整份记录。提交前核对页面选择与证据是否一致；外部流程若另存病例结论，须与页面选择保持一致。未选择时按提示补选，不为绕过必填而随意选“非目标”或“信息不足”。

### 2.1'''
text=change(text,old,new)
dictionary.write_bytes(text.replace('\n','\r\n').encode('utf-8'))

# Reuse the four existing definition cards verbatim in the HTML dictionary.
case_section=text.split('## 2. 病例级判断标签',1)[1].split('## 3. 核心肺部证据',1)[0]
cards=[]
for m in re.finditer(r'^### (2\.[1-4]) ([^\n]+)\n(.*?)(?=^### |\Z)',case_section,re.M|re.S):
    fields={k:v.strip() for k,v in re.findall(r'^- \*\*(.+?)\*\*：(.+)$',m[3],re.M)}
    cards.append({'id':m[1],'title':m[2],'group':'病例四分类','fields':fields,'entity':False,'caseDecision':True})
assert len(cards)==4
html_path=root/'documentation/layout-preview/annotation-manual-design-b.html'
html=html_path.read_text(encoding='utf-8')
items_match=re.search(r'const items=(\[.*?\]);\s*const el=',html,re.S)
assert items_match
items=json.loads(items_match[1])
assert not any(x['id'].startswith('2.') for x in items)
html=html[:items_match.start(1)]+json.dumps(cards+items,ensure_ascii=False)+html[items_match.end(1):]
html=change(html,'<div id="nav">','<div id="nav"><button class="navbtn" data-group="病例四分类">病例四分类 <span>4</span></button>')
case_renderer='''function renderCase(){
const f=current.fields;el('title').textContent=current.title;el('eyebrow').textContent='病例四分类 · 病例级单选';el('breadcrumb').textContent='病例四分类';el('lead').textContent=f['项目操作定义'];
el('meta').hidden=false;el('meta').innerHTML='<span class="chip">病例级结论</span><span class="chip neutral">必选一个 · 无默认值</span><span class="chip neutral">不框选实体</span>';
document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
const rules=section('rules','01','什么情况选择',`<div class="rule"><div class="rulebox"><h3>纳入条件</h3><p>${fmt(f['纳入标准'])}</p></div><div class="rulebox no"><h3>排除条件</h3><p>${fmt(f['排除标准'])}</p></div></div><div class="note">在Label Studio左侧标注栏顶部的“病例四分类”中选择。四个选项互斥，不需要选中实体；结论作用于整份病例，切换实体不会改变其归属。发现身份信息时停止该例，不选择、不提交。</div>`);
const examples=section('examples','02','原文与病例结论',`<div class="example"><div class="exhead"><b>正例</b></div><p>${fmt(f['正例1'])}</p></div><div class="example"><div class="exhead"><b>正例</b></div><p>${fmt(f['正例2'])}</p></div><div class="example"><div class="exhead"><b class="warn">近似反例</b></div><p>${fmt(f['近似反例'])}</p></div>`);
const reference=section('attributes','03','证据与提交要求',`<div class="scope"><span>证据要求</span><div>${fmt(f['最小完整跨度'])}</div></div><p style="margin-top:18px">结论在Label Studio中记录；支持证据、任务状态和数据问题另存外部表单。提交前核对所选结论和证据，需要更正时改选另一个选项。结论没有默认值，不得为绕过必填而随意选择。</p><div class="note">${fmt(f['常见混淆与裁决'])}</div><details><summary>定义与病例判断作用</summary><p>${fmt(f['医学定义'])}</p><p>${fmt(f['病例判断作用'])}</p></details><p style="margin-top:18px"><a href="../../annotation_agent_workflow/guides/annotation_guide_v2.1.1.html#chapter-3">查看主指南：四分类判断顺序 ↗</a></p>`);
el('body').innerHTML=tab==='examples'?examples:tab==='reference'?reference:rules+examples+reference;catalog=false;
}
'''
html=change(html,'function render(){',case_renderer+'''function render(){
const tocTitles=current.caseDecision?['选择条件','原文示例','证据与提交']:['什么情况标','原文与操作对照','属性与关系'];document.querySelectorAll('.toc a').forEach((a,i)=>{if(tocTitles[i])a.textContent=tocTitles[i]});
if(current.caseDecision){renderCase();return}''')
html=change(html,"${x.entity?'查看标签操作':'查看属性与关系'}", "${x.caseDecision?'查看病例结论':x.entity?'查看标签操作':'查看属性与关系'}")
html=change(html,'页面顶部“病例四分类”必选一个结论；证据和数据问题另存外部表单。','标注页面顶部“病例四分类”必选一个结论；证据和数据问题另存外部表单。')
html_path.write_text(html,encoding='utf-8')

preview=root/'documentation/layout-preview/annotation-manual-preview.html'
html=preview.read_text(encoding='utf-8')
case_reference='<section class="card" id="case-decision-help"><h2>病例四分类：在标注页面怎样记录</h2><p>在Label Studio左侧标注栏顶部选择一个病例结论：目标病例信号、待专业复核、非目标或信息不足。控件必选、单选、无默认值，作用于整份病例，不需要先框选或选中实体。</p><p>结论记录在Label Studio；原文证据、任务状态和数据问题仍在外部表单记录。提交前核对结论与证据，需要更正时改选其他选项。发现身份信息时停止该例，不选择、不提交。</p><p><a href="../../annotation_agent_workflow/guides/annotation_guide_v2.1.1.html#chapter-3">查看四分类判断顺序</a></p></section>'
assert '<section class="card"' in html
html=html.replace('<section class="card"',case_reference+'<section class="card"',1)
preview.write_text(html,encoding='utf-8')
print('Operating and submission instructions added; four case cards added to the HTML dictionary.')
