"""Text-only clinical clauses for learned case heads; no reference metadata."""
import re
from .clinical_evidence import project_case_cues

CUES=re.compile(r'发热|体温|咳|气促|呼吸|低氧|肺|啰音|胸|病毒|支原体|衣原体|抗原|核酸|感染|信息|病史同前|正文截断|附件|未提供')
FEATURE_PATTERNS=[r'肺炎',r'考虑.{0,12}肺炎|肺炎.{0,12}待排|不除外.{0,12}肺炎',
 r'无.{0,10}发热|否认.{0,10}发热',r'无.{0,10}咳|否认.{0,10}咳',
 r'未见.{0,12}肺炎|无.{0,12}肺炎|未见.{0,12}浸润',r'胸部CT|胸部.*CT|胸片',
 r'炎症|浸润|实变|磨玻璃|斑片',r'抗感染|抗生素|头孢|阿奇霉素',
 r'慢性|陈旧|既往|长期|纤维|条索',r'急性|加重|近日|本次|新发',
 r'心衰|肿瘤|肿块|恶性|支扩|支气管扩张|间质性',r'复诊|复查|改善|缓解|好转',
 r'未提供|同前|信息不足|不详|缺失|截断',r'主诉|现病史|入院诊断|出院诊断|病程',
 r'阳性',r'阴性',r'检验项目名称|检验结果|参考区间',r'鼻塞|流涕|咽痛|上呼吸道',
 r'建议|拟行|待查|进一步|未明确',r'无明显不适|体检|筛查']

def focus(text):
    """Keep full relevant clauses, including negation/time, and initial narrative."""
    clauses=re.split(r'(?<=[。；\n])',text)
    kept=[];seen=set()
    for i,part in enumerate(clauses):
        if i<2 or CUES.search(part):
            if part not in seen:kept.append(part);seen.add(part)
    return ''.join(kept) or text

def features(text):
    return project_case_cues(text)+[min(len(re.findall(p,text))/3.,2.) for p in FEATURE_PATTERNS]+[min(len(text)/200.,5.),float(len(text)<100)]
