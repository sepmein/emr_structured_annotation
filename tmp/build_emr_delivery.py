"""Produce task JSON and row-level triage audit without modifying the source workbook."""
import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('E:/emr_structured_annotation')
OUT=ROOT/'output/emr_data_1_20261008'
records=json.loads((OUT/'_parsed_records.json').read_text(encoding='utf-8'))
errors=json.loads((OUT/'invalid_source_rows.json').read_text(encoding='utf-8'))
decisions=json.loads((ROOT/'tmp/emr_screening_decisions.json').read_text(encoding='utf-8'))
lookup={n:k for k,ns in decisions.items() for n in ns}
assert set(lookup)=={r['row'] for r in records}
assert sum(map(len,decisions.values()))==len(lookup)

LABELS={'core':'优先保留','boundary':'边界及对照保留','pending':'信息不足待补充','exclude':'明确无关剔除','invalid':'源JSON截断隔离'}
CORE_REASONS={
7:'急性咳嗽咳痰、发热及支扩感染，有呼吸道病原检测；不把支扩等同肺炎。',
13:'发热咳嗽、甲流结果及右下肺炎性病变，属于本项目直接证据。',
15:'发热咳痰、异地就诊肺炎诊断及旅居信息；不同文书对起病经过有差异。',
16:'虽有换药就诊模板，另含咳嗽咳痰伴发热；保留并核对是否错拼。',
17:'发热呼吸困难、两肺炎症；同时引用不同日期影像，需核对索引事件。',
18:'发热咳嗽、甲流阳性和两肺下叶炎症；有完整治疗及转归。',
26:'反复发热咳嗽，影像右肺炎症，肺炎收住院。',
29:'明确肺部感染治疗复诊，仍偶有咳嗽。',
30:'发热咳嗽、两肺炎症和部分实变；混入旧球形肺炎影像需核对。',
33:'急性发热、咳嗽咳痰，甲型流感抗原阳性；不自动判肺炎。',
38:'住院过程两肺炎症，痰培养耐药肺克及治疗转归，不能仅按肾病主诉剔除。',
39:'咳嗽咳痰伴发热，支扩伴周围感染及好转过程。',
40:'心血管主诉之外明写肺部感染、咳嗽咳痰，不能按主诉直接剔除。',
44:'当前咳嗽咳痰伴双肺湿啰音，可支持专业复核入口。',
50:'持续发热、肺部炎症及左下肺实变，肺炎收住院。',
53:'发热伴咳痰，胸部CT两肺散在感染。',
60:'儿童发热咳嗽、右下肺炎症及肺炎诊断，含病原阴性与治疗过程。',
61:'发热伴咳嗽及呼吸道病原检查；部分结果字段填成项目名称，需复核。',
71:'发热、咳嗽黄痰及指氧93%，有病原检查计划。',
73:'发热咳嗽、左下肺炎症伴实变及疗程转归。',
79:'儿童发热咳嗽连续复诊，含时间与病程变化，优先保留。',
87:'本次再次发热咳痰，湿啰音及空气下SpO2 92%，适合边界专业复核。',
88:'左下肺支扩伴周围炎症且影像进展，保留核对感染活动性。',
90:'两肺炎症进展的影像证据，不能被同条其他脏器报告掩盖。',
91:'肺炎检查背景、右下肺炎症进展及部分吸收的连续影像。',
95:'原文明写近期反复肺部感染及病原信息。',
98:'反复发热咳嗽、胸部CT两肺炎症及出院病程。',
103:'儿童发热咳嗽复诊，保留治疗前后病程变化。',
147:'儿童咳嗽加重伴低热，治疗后复诊，保留当前与既往状态区别。',
162:'两肺炎症进展、左下肺实变，发热背景。',
170:'复诊病种明写肺部感染，需核对是否同一连续疗程。',
173:'肺炎背景下两肺炎症、右中叶实变且进展。',
204:'检查申请症状写肺炎，虽然报告仅胆囊术后，仍需核对申请诊断。',
244:'当前咳嗽咳痰伴双肺湿啰音，可支持专业复核入口。',
250:'明确肺部感染复诊用药，保留连续诊疗线索。',
270:'肺炎继续用药，伴左下肺湿性啰音。',
284:'肺炎及衣原体感染检查背景，右肺炎症吸收的转归影像。',
292:'咳嗽咳痰并发热，继续治疗。',
317:'急性发热伴咳嗽，保留专业复核候选。'
}
assert set(CORE_REASONS)==set(decisions['core'])

GROUPS=[
([4,14,31,37,85,94,111,134,239,253], '肺部影像边界/阴性对照', '含肺部影像证据或阴性胸部CT，可训练慢性改变、肿瘤/心衰解释与感染的区分；保留不等于目标病例。'),
([8,28,64,75,76,77,78,89,93,99,113,117,184], '病原/检验边界', '检验或病原相关记录，缺少充分临床或标本背景；保留作检测、否定或结果错配复核，不据项目名或代码推断肺炎。'),
([20,67,82,106,133,141,149,164,212,321], '非感染解释及胸闷气促对照', '含胸闷、气短、心肺检查或相关既往症状；需区分心源性/血管性解释，不能直接认定感染。'),
([27,54,55,81,127,136,160,174,178,181,182,194,197,198,211,222,223,226,236,290], '慢性肺病/复诊边界', '含慢性肺病、支扩、肺结核或呼吸病复诊，保留稳定、持续、改善与本次急性加重的区别。'),
([2,5,56,62,74,83,137], '发热及感染解释边界', '存在发热或相关病原/肺部线索，但未达到明确肺炎证据；保留排除、替代解释和补充复核用途。'),
([105,167,168,169,176,177,213,217,240,241,256,264,291,320], '鼻咽轻症对照', '仅鼻咽/上呼吸道线索，当前不支持肺炎；宽口径初筛保留作轻症非目标对照，缩小规模时可优先移出。'),
([9,11,42,52,70,86,92,96,102,107,108,120,121,125,126,129,130,131,132,145,154,157,199,202,203,216,224,225,227,228,229,242,249,255,257,262,268,274,289,300,304,318,326], '呼吸症状/疗程对照', '存在咳嗽咳痰、呼吸道感染或相关疗程线索；即使症状轻、已好转或目前无发热，也不作为完全无关删除。')
]
boundary_map={n:(g,r) for ns,g,r in GROUPS for n in ns}
assert set(boundary_map)==set(decisions['boundary']), (set(decisions['boundary'])-set(boundary_map),set(boundary_map)-set(decisions['boundary']))

SENSITIVE_KEYS={'id_card','patient_name','patient_id','serial_number','patient_phone','phone','telephone','mobile','address','patient_address','contact_name','contact_phone'}
name_patterns=[
 re.compile(r'(?:患者姓名|姓名)\s*[：:]\s*([\u4e00-\u9fff]{2,4})(?=[\s，,；;]|年龄|性别|科室)'),
 re.compile(r'患者([\u4e00-\u9fff]{2,4})(?=[，,]\s*[男女](?:[，,]|性))'),
 re.compile(r'患者([\u4e00-\u9fff]{2,4})(?=[，,]\s*因[“"\'])')
]
names=set()
for x in records:
 for pat in name_patterns:
  names.update(pat.findall(x['data']['text']))
 # Explicit names from the observed first-course documentation.
 v=x['data'].get('patient_name','')
 if isinstance(v,str) and re.fullmatch(r'[\u4e00-\u9fff]{2,4}',v): names.add(v)
names-= {'目前','自述','此次','自发病','入院','既往','现为','现无','为求','神清','无明显'}

def scrub(s):
 for name in sorted(names,key=len,reverse=True): s=s.replace(name,'[姓名已遮盖]')
 s=re.sub(r'(?<!\d)\d{17}[\dXx](?!\d)','[证件号已遮盖]',s)
 s=re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)','[电话已遮盖]',s)
 s=re.sub(r'((?:住院号|门诊号|床位号|床号)\s*[：:]\s*)[A-Za-z0-9-]+',r'\1[已遮盖]',s)
 return s

def clean_obj(v):
 if isinstance(v,dict): return {k:clean_obj(val) for k,val in v.items() if k.lower() not in SENSITIVE_KEYS}
 if isinstance(v,list): return [clean_obj(z) for z in v]
 if isinstance(v,str): return scrub(v)
 return v

def pseudo(s): return 'P-'+hashlib.sha256(str(s).encode()).hexdigest()[:12]

def normalize(t):
 t=re.sub(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?','',t)
 return re.sub(r'\s+','',t)

norm_groups=defaultdict(list)
for x in records: norm_groups[normalize(x['data']['text'])].append(x['row'])

def excerpt(t,kind):
 lines=[z.strip() for z in t.splitlines() if z.strip() and not re.match(r'(操作时间|诊断时间|确定诊断日期|院内检验项目代码|标准化检验定性结果代码)',z)]
 pat=r'肺炎|肺部感染|炎症|实变|发热|咳嗽|咳痰|气促|气急|甲型流感|肺炎支原体|呼吸道|未见明显异常'
 if kind=='exclude':
  chosen=[z for z in lines if z.startswith(('主诉','现病史','症状描述','检验报告结果-客观提示'))][:2]
 else:
  chosen=[z for z in lines if re.search(pat,z)][:3]
 if not chosen: chosen=lines[:3]
 return scrub('；'.join(z[:100] for z in chosen))[:300]

tasks={k:[] for k in decisions}
audits=[]
redactions=[]
for x in records:
 row,d=x['row'],x['data']; kind=lookup[row]; text=d['text']; patient=pseudo(d['patient_id'])
 cleaned=clean_obj(d)
 cleaned['patient_id']=patient
 cleaned['patient_name']='已去除直接身份字段'
 cleaned['serial_number']=f'R-{row:04d}'
 cleaned['text']=scrub(text)
 cleaned['chief_complaint_text']=cleaned['text']
 cleaned['text_length']=len(cleaned['text'])
 cleaned['source_row']=row
 cleaned['source_sheet']=x['sheet']
 cleaned['screening_status']=LABELS[kind]
 cleaned['screening_is_annotation']=False
 tasks[kind].append({'data':cleaned})
 if kind=='core': group='肺部证据/急性呼吸信号'; reason=CORE_REASONS[row]
 elif kind=='boundary': group,reason=boundary_map[row]
 elif kind=='pending':
  group='缺少病种、前文或可解释检查'
  reason='仅有同前、复诊、配药、占位文本或无归属检验，无法确认所指病种；待补充，不作无关剔除。'
 else:
  group='非呼吸索引事件'
  cc=re.findall(r'(?:主诉|症状描述)[：:][^\n]+',text)
  topic=cc[0][:45] if cc else next((z[:45] for z in text.splitlines() if z.startswith(('现病史','检验报告结果-客观提示','院内检验项目名称','辅助检查'))),'本条非呼吸记录')
  reason=f'{topic}；本条未见独立肺炎、急性呼吸或本项目病原证据，仅常规阴性模板不作为纳入依据。'
  if row in (47,140): reason='妇科索引事件中的支原体/解脲支原体线索，非肺炎支原体证据；未见独立呼吸信号。'
  if row in (80,115): reason='仅消化道或妇科/尿液常规微生物信息，无本项目呼吸病原或临床线索。'
  if row==313: reason='仅建议电子喉镜，无肺炎触发或其他可用呼吸感染信息。'
 q=[]
 if len(norm_groups[normalize(text)])>1: q.append('去日期及空白后同文：'+','.join(map(str,norm_groups[normalize(text)])))
 if row in (16,17,30,79): q.append('多文书起病/日期/索引归属需核对')
 if row==61: q.append('定性结果名称疑似填成项目名称')
 if row==78: q.append('未生长/未检出文字与阳性代码冲突；按原文保留，不推断阳性')
 if row in (178,329): q.append('同条症状叙述或字段格式存在矛盾，需回源')
 name_hit=any(n in text for n in names)
 if name_hit: q.append('显式姓名已自动遮盖；仍需脱敏验收')
 if re.search(r'姓名|证件|身份证|电话|手机|住址|地址',scrub(text)): q.append('有身份/地址标题，需确认是否仍有残留')
 group_members=norm_groups[normalize(text)]
 audits.append({'源Excel行号':row,'工作表':x['sheet'],'患者代号':patient,'筛选结论':LABELS[kind],'用途分组':group,'原文依据':excerpt(text,kind),'逐条筛选理由':reason,'数据问题':';'.join(q),'原文本字符数':len(text),'推荐导入':kind in ('core','boundary'),'候选保留':kind!='exclude'})
 if cleaned['text']!=text: redactions.append({'source_row':row,'original_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'export_text_sha256':hashlib.sha256(cleaned['text'].encode()).hexdigest(),'original_length':len(text),'export_length':len(cleaned['text'])})

for e in errors:
 audits.append({'源Excel行号':e['row'],'工作表':e['sheet'],'患者代号':'','筛选结论':LABELS['invalid'],'用途分组':'源单元格长度上限','原文依据':'','逐条筛选理由':'单元格32767字符，JSON字符串中途截断；保留原始单元格等待回源补齐。','数据问题':e['error'],'原文本字符数':None,'推荐导入':False,'候选保留':False})
audits.sort(key=lambda a:a['源Excel行号'])

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
dump('label_studio_recommended_156.json',sorted(tasks['core']+tasks['boundary'],key=lambda t:t['data']['source_row']))
dump('label_studio_priority_39.json',tasks['core'])
dump('label_studio_candidates_194.json',sorted(tasks['core']+tasks['boundary']+tasks['pending'],key=lambda t:t['data']['source_row']))
dump('label_studio_pending_38.json',tasks['pending'])
dump('label_studio_all_valid_327.json',sorted(sum(tasks.values(),[]),key=lambda t:t['data']['source_row']))
dump('screening_audit.json',audits)
dump('redaction_log.json',redactions)
with (OUT/'screening_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
summary={'source_rows':329,'valid_json':327,'truncated_json_rows':[35,36],'counts':{LABELS[k]:len(v) for k,v in decisions.items()},'recommended_import':156,'candidate_pool_including_pending':194,'pending':38,'exclude':133,'unique_patients_valid':len({x['data']['patient_id'] for x in records}),'unique_patients_recommended':len({t['data']['patient_id'] for t in tasks['core']+tasks['boundary']}),'unique_patients_candidates':len({t['data']['patient_id'] for t in tasks['core']+tasks['boundary']+tasks['pending']}),'date_normalized_duplicate_groups':[v for v in norm_groups.values() if len(v)>1],'text_redacted_rows':len(redactions),'source_sha256':hashlib.sha256(Path('D:/OneDrive/03_Resource/Documents/Desktop/emr_data_1.xlsx').read_bytes()).hexdigest(),'method':'逐条辅助初筛与语义核对。长文本按主诉、现病史及全文检索的相关片段复核；并非临床专家逐字双标，也不是四分类金标准。','scope':'宽口径候选保留，包含轻症、稳定慢性肺病、阴性和非感染解释对照；信息不足单列。','privacy':'导出移除直接身份字段，患者号换成稳定SHA256派生代号，自动遮盖识别到的姓名、完整证件号及手机号；不是完整去标识化认证，人工标注前需验收。'}
dump('screening_summary.json',summary)
assert len(audits)==329 and {a['源Excel行号'] for a in audits}==set(range(1,330))
for name,n in [('label_studio_recommended_156.json',156),('label_studio_candidates_194.json',194),('label_studio_priority_39.json',39),('label_studio_pending_38.json',38),('label_studio_all_valid_327.json',327)]:
 arr=json.loads((OUT/name).read_text(encoding='utf-8')); assert len(arr)==n
 assert len({t['data']['source_row'] for t in arr})==n
 assert all(t['data']['text']==t['data']['chief_complaint_text'] and len(t['data']['text'])==t['data']['text_length'] for t in arr)
 assert all('annotations' not in t and 'predictions' not in t for t in arr)
print(json.dumps({k:v for k,v in summary.items() if k not in ('date_normalized_duplicate_groups','source_sha256')},ensure_ascii=False,indent=2))
