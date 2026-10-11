"""Build traceable AI sample adjudication proposals; never expert gold.

Case decisions and exceptions are explicitly supplied by a reviewer. Literal
rules propose structural points, with provenance and unresolved items retained.
Imports are stdlib-only and never send medical text to an external service.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import html
import json
import re

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.annotation_analysis.double_annotations import parse_annotation, safe_text
from emr_annotation.training_data.schema_multitask import sha256, write_json, relation_type_allowed


PATTERNS = {
    '发热': r'发热|高热|低热|发烧', '鼻翼煽动': r'鼻翼(?:煽动|扇动)|鼻翼无(?:煽动|扇动)',
    '呼吸衰竭': r'(?:[ⅠⅡI]+型)?呼吸衰竭', '三凹征': r'三凹征',
    '喘鸣喘息': r'喘鸣|喘息|哮鸣音', '意识障碍': r'意识障碍|意识模糊|神志不清|意识不清|昏迷',
    '惊厥': r'惊厥|抽搐', '拒食或喂养困难': r'拒食|拒奶|喂养困难|不能饮水|吸吮无力',
    '脱水征': r'(?:轻度|中度|重度)?脱水(?:征|貌)?',
    '休克': r'(?:感染性|脓毒性|心源性|低血容量性)?休克',
    '器官衰竭': r'(?:急性|慢性)?(?:多器官功能衰竭|多器官衰竭|心力衰竭|肾衰竭|肝衰竭)',
    '机械通气': r'(?:有创|无创)?机械通气|(?:有创|无创)通气|呼吸机辅助通气',
    '新冠病毒': r'新型冠状病毒|新冠病毒|SARS-CoV-2',
    '流感病毒': r'(?<!副)(?:甲型|乙型|甲|乙|A型|B型)?(?:流感病毒|流行性感冒病毒)|(?:甲型|乙型)流感(?=抗原|检测)|甲流|乙流',
    '呼吸道合胞病毒': r'呼吸道合[胞孢]病毒', '腺病毒': r'腺病毒',
    '人偏肺病毒': r'(?:人)?偏肺病毒|HMPV', '副流感病毒': r'副流感病毒',
    '普通冠状病毒': r'(?<!新型)冠状病毒(?:229E/NL63型|OC43型|229E|NL63|OC43|HKU1)',
    '博卡病毒': r'(?:人)?博卡病毒', '鼻病毒': r'(?:人)?鼻病毒',
    '肠道病毒': r'肠道病毒|柯萨奇病毒|EV-71-IgM肠病毒', 'A族链球菌': r'A(?:族|组)链球菌',
    '百日咳鲍特菌': r'百日咳(?:鲍特|杆)菌', '肺炎链球菌': r'肺炎链球菌',
    '流感嗜血杆菌': r'流感嗜血杆菌', '军团菌': r'(?:嗜肺)?军团菌',
    '肺炎克雷伯菌': r'肺炎克雷伯(?:氏)?菌|肺克', '肺炎支原体': r'肺炎支原体',
    '曲霉菌': r'(?:烟|黄)?曲霉(?:菌)?', '隐球菌': r'(?:新型)?隐球菌',
    '鹦鹉热衣原体': r'鹦鹉热衣原体', '肺炎衣原体': r'肺炎衣原体',
}
LUNG = re.compile(r'(?:社区获得性|支气管|大叶性|细菌性|病毒性|间质性)?肺炎(?!支原体|衣原体|克雷伯|链球)|(?:左|右|双|两)?(?:上|下)?肺(?:上叶|中叶|下叶)?(?:散在|少许|小片)*(?:炎症|炎性病变|炎性渗出|感染|实变)|肺部(?:炎症|感染)')
TIME = re.compile(r'\d{4}[-./年]\d{1,2}[-./月]\d{1,2}日?|\d{1,2}月\d{1,2}日|(?:近|入院前)?\d+(?:-\d+)?(?:余)?(?:天|日|周|月|年|小时)(?:前|后)?|今日|今天|昨日|昨天|昨晚|今晨|近期|今')
MEASURE = re.compile(r'(?P<indicator>体温|热峰|(?<![A-Za-z])T(?![A-Za-z])|呼吸频率|呼吸|(?<![A-Za-z])RR?(?![A-Za-z])|PaO2|FiO2|吸氧浓度|SpO2|静息(?:状态)?指氧饱和度|指氧饱和度|血氧饱和度|指末氧|氧饱和度)\s*[:：]?\s*(?:最高可达|最高|可达|升高至|下降至)?\s*(?P<compare>[<>≤≥＜＞])?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>℃|°C|摄氏度|度|百分比|%|次[/／](?:分钟|分|min)|mmHg)?', re.I)


def segment(text, start, end):
    """Punctuation-bounded clause plus exact evidence coordinates."""
    lo = max((text.rfind(c, 0, start) for c in '\n。；;，,'), default=-1) + 1
    ends = [text.find(c, end) for c in '\n。；;，,']
    hi = min((e for e in ends if e >= 0), default=len(text))
    return lo, hi, text[lo:hi]


def clinical_attributes(text, ent, schema):
    s, e = ent['start'], ent['end']
    lo, hi, clause = segment(text, s, e)
    prefix, suffix = text[lo:s], text[e:hi]
    # Comma-separated lists can share a negation; never cross a sentence.
    sentence_lo = max(text.rfind('\n', 0, s), text.rfind('。', 0, s), text.rfind('；', 0, s)) + 1
    broad = text[sentence_lo:s]
    if ent['label'] in ('肺炎影像','肺炎诊断'):
        # Prior symptom negations do not govern a subsequent objective report.
        markers_source = list(re.finditer(r'CT[:：示]?|胸片[:：示]?|检查结论|诊断|外院诊断', broad, re.I))
        if markers_source:
            broad = broad[markers_source[-1].end():]
    markers = list(re.finditer(r'无(?!明显诱因|诱因)|未见|否认|不伴|未闻及|未诉|有|伴|出现', broad))
    shared_absent = bool(markers and markers[-1].group() in ('无', '未见', '否认', '不伴', '未闻及', '未诉'))
    if re.search(r'不除外|疑似|考虑|待排|可能', prefix[-12:] + suffix[:8]):
        context = 'possible'
    elif shared_absent or '无' in text[s:e] or re.match(r'\s*(?:未闻及|阴性)', suffix):
        context = 'known_absent'
    else:
        context = 'known_present'
    hypothetical = bool(re.search(r'拟|计划|建议.*(?:排查|检查)|若|如出现|随时可能出现', prefix[-25:]))
    if re.search(r'收(?:住|治)?入院|收入.*病房', suffix):
        hypothetical = False
    if hypothetical:
        context = 'unknown'
    line_lo = text.rfind('\n', 0, s) + 1
    line_prefix = text[line_lo:s]
    temporal = 'future' if hypothetical else 'historical' if re.search(r'\d+(?:天|日|周|月|年)前|既往|曾|往年', prefix) or line_prefix.startswith('既往史：') else 'current'
    subject = 'family' if re.search(r'母亲|父亲', prefix) else 'other' if re.search(r'同学|朋友|接触过.*患者', prefix) else 'patient'
    values = {'finding_context': context, 'temporality': temporal,
              'conditional': 'yes' if re.search(r'活动后|若|如出现', prefix) else 'no',
              'hypothetical': 'yes' if hypothetical else 'no', 'experiencer': subject}
    if context != 'known_absent':
        course = ('worsening' if re.search(r'加重|进展|恶化', suffix) else
                  'improving' if re.search(r'好转|改善|吸收|减少', suffix) else
                  'resolved' if re.search(r'已缓解|消失', suffix) else None)
        if course:
            values['clinical_course'] = course
    return {k: v for k, v in values.items() if ent['label'] in schema['fields'][k]['scope']}, (lo, hi)


def lab_attributes(text, ent):
    s, e = ent['start'], ent['end']
    line_lo = text.rfind('\n', 0, s) + 1
    line_hi = text.find('\n', e)
    line_hi = len(text) if line_hi < 0 else line_hi
    if '院内检验项目名称' in text[line_lo:line_hi]:
        next_block = text.find('操作时间：', e)
        hi = next_block if next_block >= 0 else len(text)
        block = text[line_lo:hi]
    else:
        lo, hi, block = segment(text, s, e)
        # A result after a semicolon belongs to another test group, never borrow.
        line_lo = lo
    tested = bool(re.search(r'检测|测定|Ig[MＧG]|抗体|核酸|抗原|培养|阳性|阴性|检出', block, re.I))
    if not tested:
        return {}, (line_lo, hi), []
    suffix = re.sub(r'^.*(?:参考区间|项目代码|定性结果代码|定量结果超出).*$','',text[e:hi],flags=re.M)
    result = ('unknown' if re.search(r'可疑阳性|临界|待回报|拟.*检测', block) else
              'negative' if re.search(r'阴性|未检出', suffix) else
              'positive' if re.search(r'阳性|检出', suffix) else 'unknown')
    method = ('抗体IgM' if re.search('[Il]gM', block, re.I) else '抗体IgG' if re.search('[Il]gG', block, re.I) else
              'mNGS' if re.search(r'mNGS|宏基因组', block, re.I) else
              '核酸检测' if re.search(r'核酸|PCR|DNA|RNA', block, re.I) else
              '抗原快速检测' if '抗原' in block else '培养' if '培养' in block else '原文未说明')
    specimen = ('口咽/鼻咽拭子' if re.search(r'鼻咽|口咽|咽拭子', block) else
                '肺泡灌洗液' if re.search(r'肺泡灌洗|BALF', block) else
                '气管吸取物' if '气管吸取' in block else '深部痰液' if '深部痰' in block else
                '尿' if re.search(r'尿培养|尿标本', block) else
                '痰' if re.search(r'痰培养|痰标本|【痰】', block) else '原文未说明')
    conflicts = ['lab_result_conflict'] if '阴性' in suffix and '阳性' in suffix else []
    if conflicts:
        result = 'unknown'
    return {'test_result': result, 'specimen_type': specimen, 'detection_method': method}, (line_lo, hi), conflicts


def propose(task, schema):
    """Literal proposals with explicit reason. No hidden medical classifier."""
    text = task['data']['text']
    entities, links, issues = {}, set(), []

    def add(start, end, label, reason):
        key = (start, end, label)
        entities.setdefault(key, {'start': start, 'end': end, 'label': label,
                                 'text': text[start:end], 'attributes': {}, 'reason': reason,
                                 'rule_ref': 'guide:5-9;dictionary:3-12'})
        return key

    for label, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, re.I):
            if label == '肺炎克雷伯菌' and match.group() == '肺克' and not re.search(r'肺炎克雷伯(?:氏)?菌', segment(text, *match.span())[2]):
                issues.append({'code':'unconfirmed_pathogen_abbreviation','start':match.start(),'end':match.end(), 'reason':'简称未在同组明确定义，待复核，不直接映射菌种。'})
                continue
            if label == '发热' and text[match.end():match.end()+2] == '门诊':
                continue
            if label == '器官衰竭' and '鉴别诊断:' in text[max(0, match.start()-1500):match.start()]:
                issues.append({'code':'educational_differential_not_patient_fact','start':match.start(),'end':match.end(),'reason':'鉴别诊断介绍一般疾病过程，不当患者已发生事实。'})
                continue
            add(*match.span(), label, '全文逐标签字面扫描；该词符合操作定义，属性另按所属语境复核。')
    for m in LUNG.finditer(text):
        lo, hi, clause = segment(text, *m.span())
        line_lo = text.rfind('\n', 0, m.start()) + 1
        prefix = text[line_lo:m.start()]
        label = '肺炎影像' if re.search(r'CT|胸片|影像|检查结论|客观(?:所见|提示)', prefix, re.I) else '肺炎诊断'
        if re.search(r'诊断(?:依据)?[:：]?[“"]?\s*$', prefix) or re.match(r'[”"]?收(?:住|治)?入院|[”"]?收入院',text[m.end():hi]):
            label = '肺炎诊断'
        if re.search(r'慢性|稳定.*间质性|非感染性', clause):
            issues.append({'code': 'noninfectious_or_chronic_lung_mention', 'start': m.start(), 'end': m.end(),
                           'reason': '慢性或非感染性肺部改变不机械映射感染性肺炎实体。'})
            continue
        add(*m.span(), label, '明确肺部概念，影像/临床按原文来源区分；病例纳入另审索引归属。')
    for m in MEASURE.finditer(text):
        indicator = m.group('indicator').lower()
        label = ('体温' if indicator in ('体温', '热峰', 't') else '呼吸频率' if indicator in ('r', 'rr', '呼吸', '呼吸频率') else
                 '动脉氧分压(PaO2)' if indicator == 'pao2' else '吸氧浓度(FiO2)' if indicator in ('fio2', '吸氧浓度') else '静息状态指氧饱和度')
        if label == '静息状态指氧饱和度' and re.search(r'运动后|步行后|步行试验|负荷后', segment(text, *m.span())[2][:m.start()-segment(text, *m.span())[0]]):
            continue
        if label == '呼吸频率' and (m.group('unit') or '').lower() not in ('次/分', '次／分', '次/分钟', '次/min'):
            continue
        a = add(*m.span('indicator'), label, '明确生命体征指标词；不把数值或单位重复标作指标。')
        for group, component in (('value', '数值'), ('unit', '单位'), ('compare', '比较符')):
            if m.group(group):
                b = add(*m.span(group), component, '同一测量组中原文明写的组成；不补单位或阈值。')
                links.add((a, b, '测量'))
    # Raw exposure regions are only candidates when they contain an event, not a place/name.
    parsed = [parse_annotation(task, ann, schema, 'raw', f"p{task['project']}_t{task['id']}") for ann in task['annotations']]
    for p in parsed:
        for ent in p['entities']:
            label = ent['label']
            if label not in schema['controls']['epidemics_labels']:
                continue
            s, e = ent['start'], ent['end']
            if not isinstance(s, int) or not isinstance(e, int) or not 0 <= s < e <= len(text):
                continue
            phrase = text[s:e]
            valid = (bool(re.search(r'旅游|旅行|旅居|赴|去.*(?:探望|游)|外出', phrase)) if label == '境内外旅居史' else
                     bool(re.search(r'多名|多人|两人|数人', phrase)) if label == '周围聚集性发病现象' else
                     bool(re.search(r'接触.*(?:禽|鸟|动物)|(?:禽|鸟|动物).*接触', phrase)) if label == '可疑动物或动物制品接触史' else
                     bool(re.search(r'养殖|禽类|病原.*实验室', phrase)) if label == '高风险职业史' else
                     bool(re.search(r'鸟粪|禽类污染|疫水|疫区', phrase)))
            # Unfilled epidemic-history prompts are templates, not facts.
            if valid and not (re.search(r'史[：:]?\s*[。；]?$', phrase) and not re.search(r'无|未|否认|有|旅游|探望', phrase)):
                add(s, e, label, '原始暴露候选含明确事件，未将单个朋友或孤立地点升级为暴露。')
            else:
                issues.append({'code': 'unsupported_exposure_region', 'start': s, 'end': e,
                               'reason': '原始标签不满足该暴露操作定义，未沿用。'})
    # Explicit event-local times; do not connect administrative dates to all events.
    event_keys = [k for k in entities if k[2] not in set(schema['controls']['measure_labels']) | set(schema['controls']['measure_entities'])]
    for a in event_keys:
        s, e, label = a
        lo, hi, clause = segment(text, s, e)
        times = list(TIME.finditer(clause))
        for m in times:
            ts, te = lo + m.start(), lo + m.end()
            if ts < e and te > s:
                continue
            # Review explicit single event clauses only; shared or remote scopes stay unknown.
            local_events = [k for k in event_keys if lo <= k[0] < hi]
            if len(local_events) != 1:
                continue
            between = text[min(e, te):max(s, ts)]
            if len(between) > 18:
                continue
            value = m.group()
            if ts >= e and re.fullmatch(r'\d+(?:-\d+)?(?:余)?(?:天|日|周|月|年|小时)', value):
                kind = '持续时长'
            elif re.search(r'开始|起病', between):
                kind = '起始时间'
            elif re.search(r'退热|消失|结束|停止', between):
                kind = '结束时间'
            else:
                kind = '发生时间'
            b = add(ts, te, '时间表达', '同一明确事件子句的原文时间；未按最近距离跨句配对。')
            links.add((a, b, kind))
    # Lab export block dates belong to that individual test; precise ownership is encoded.
    for a in event_keys:
        if a[2] not in {l for g in ('pathogen', 'bacteria', 'other_pathogen') for l in schema['controls'][g]}:
            continue
        s, e, _ = a
        line_lo = text.rfind('\n', 0, s) + 1
        if '院内检验项目名称' not in text[line_lo:s]:
            continue
        block_lo = text.rfind('操作时间：', 0, s)
        date = TIME.search(text, block_lo, s) if block_lo >= 0 else None
        if date:
            b = add(*date.span(), '时间表达', '独立检验记录组的操作日期，作为该检测的记录时点。')
            links.add((a, b, '发生时间'))
    for key, ent in entities.items():
        if ent['label'] in schema['fields']['test_result']['scope']:
            attrs, ev, flags = lab_attributes(text, ent)
            issues.extend({'code': f, 'start': ent['start'], 'end': ent['end'], 'reason': '同组检验原文存在冲突，结果保留unknown。'} for f in flags)
        else:
            attrs, ev = clinical_attributes(text, ent, schema)
        ent['attributes'] = attrs
        ent['attribute_evidence'] = {'start': ev[0], 'end': ev[1]}
    values = sorted(entities.values(), key=lambda x: (x['start'], x['end'], x['label']))
    ids = {(ent['start'], ent['end'], ent['label']): f'e{i}' for i, ent in enumerate(values)}
    for ent in values:
        ent['id'] = ids[(ent['start'], ent['end'], ent['label'])]
    relations = [{'from_id': ids[a], 'to_id': ids[b], 'type': kind, 'direction': 'right',
                  'reason': '明确同组端点及语义归属；样例关系仍待人工确认。', 'rule_ref': 'guide:9.3'}
                 for a, b, kind in sorted(links) if a in ids and b in ids and relation_type_allowed(a[2], b[2], kind, schema)]
    return values, relations, issues, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--overrides', type=Path, help='Output-controlled point corrections keyed by source index')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    schema = load_double_annotation_schema(args.schema)
    tasks = json.loads(args.tasks.read_text('utf-8'))
    notes = json.loads(args.notes.read_text('utf-8')) if args.notes else {}
    overrides = json.loads(args.overrides.read_text('utf-8')) if args.overrides else {}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index, task in enumerate(tasks):
        entities, relations, issues, parsed = propose(task, schema)
        key = f"p{task['project']}_t{task['id']}"
        note = notes.get(str(index))
        sample = {'task_id': key, 'text': task['data']['text'], 'project': task['project'],
                        'source_task_id': task['id'], 'source_index': index,
                        'patient_id': str(task['data'].get('patient_group_id') or task['data'].get('patient_id')),
                        'text_sha256': hashlib.sha256(task['data']['text'].encode()).hexdigest(),
                        'entities': entities, 'relations': relations, 'issues': issues,
                        'case_decision': note[0] if note else None,
                        'case_rule': note[1] if note else '', 'case_reason': note[2] if note else '',
                        'case_reviewed_by_ai': bool(note), 'reference_status': 'ai_sample_not_expert_gold',
                        'point_review_method': 'literal_rule_proposal_with_case_level_AI_review_and_exceptions',
                        'ner_supervised': True, 'relations_supervised': bool(relations),
                        'relations_supervision': 'positive_only', 'expert_signed': False}
        apply_overrides(sample, overrides.get(str(index), {}), schema)
        samples.append(sample)
    ledger, counts = build_point_ledger(tasks, samples, schema)
    validate_samples(samples, schema)
    write_json(args.output_dir / 'proposals.json', samples)
    write_json(args.output_dir / 'point_decisions.json', ledger)
    write_json(args.output_dir / 'point_summary.json', counts)
    write_json(args.output_dir / 'label_coverage.json', [
        {'task_id': s['task_id'], 'label': label,
         'sample_mentions': sum(e['label'] == label for e in s['entities']),
         'method': 'literal_scan_and_original_point_reconciliation', 'expert_exhaustive_review': False}
        for s in samples for label in schema['labels']])
    write_json(args.output_dir / 'label_studio_ai_samples.json', to_label_studio(samples, schema))
    (args.output_dir / 'review.html').write_text(build_review_html(samples, ledger, counts), 'utf-8')
    write_json(args.output_dir / 'manifest.json', {'status': 'ai_sample_proposals',
        'tasks': len(samples), 'cases_with_AI_notes': len(notes), 'expert_signed': 0,
        'inputs': {str(p): sha256(p) for p in (args.tasks, args.schema) + ((args.notes,) if args.notes else ()) + ((args.overrides,) if args.overrides else ())},
        'code_sha256': sha256(__file__), 'created_at': datetime.now().astimezone().isoformat()})
    print('PROPOSALS', len(samples), 'entities', sum(len(r['entities']) for r in samples), 'relations', sum(len(r['relations']) for r in samples))


def apply_overrides(sample, patch, schema):
    """Explicit corrections; preserve IDs, evidence offsets and every action."""
    sample['point_overrides'] = patch
    for rule in patch.get('entities', []):
        for ent in sample['entities']:
            if all(ent.get(k) == rule[k] for k in ('start', 'end', 'label', 'text') if k in rule):
                ent['attributes'].update(rule.get('attributes', {}))
                if 'set_label' in rule:
                    ent['label'] = rule['set_label']
                ent['reason'] = rule['reason']
                if rule.get('remove'):
                    ent['removed'] = True
    sample['entities'] = [e for e in sample['entities'] if not e.pop('removed', False)]
    ids = {e['id'] for e in sample['entities']}
    sample['relations'] = [r for r in sample['relations'] if r['from_id'] in ids and r['to_id'] in ids]
    if patch.get('ner_supervised') is False:
        sample['ner_supervised'] = False
    sample['relations_supervised'] = bool(sample['relations'])


def validate_samples(samples, schema):
    for s in samples:
        if s['case_decision'] not in schema['fields']['case_decision']['values']:
            raise ValueError('Unreviewed case: ' + s['task_id'])
        ids = {}
        for e in s['entities']:
            if e['id'] in ids or e['label'] not in schema['labels'] or not (0 <= e['start'] < e['end'] <= len(s['text'])):
                raise ValueError('Invalid entity: ' + s['task_id'])
            if e['text'] != s['text'][e['start']:e['end']]:
                raise ValueError('Text mismatch')
            ids[e['id']] = e
            for f, v in e['attributes'].items():
                rule = schema['fields'][f]
                if e['label'] not in rule['scope'] or v not in rule['values']:
                    raise ValueError('Invalid attribute: ' + f)
            for f, rule in schema['fields'].items():
                if rule['per_region'] and rule['required'] and e['label'] in rule['scope'] and f not in e['attributes']:
                    raise ValueError('Missing required attribute: ' + f)
        for r in s['relations']:
            if r['from_id'] not in ids or r['to_id'] not in ids or not relation_type_allowed(ids[r['from_id']]['label'], ids[r['to_id']]['label'], r['type'], schema):
                raise ValueError('Invalid relation')


def build_point_ledger(tasks, samples, schema):
    """Reconcile every source result. Unknown points stay pending, never gold."""
    ledger = []
    raw_count = 0
    for task, s in zip(tasks, samples):
        referenced = set()
        for ann in task['annotations']:
            p = parse_annotation(task, ann, schema, 'a' + str(ann['id']), s['task_id'])
            old_ids = {}
            for e in p['entities']:
                old_ids.setdefault(e['id'], []).append(e)
            for raw in p['inventory']:
                raw_count += 1
                status, final, reason, rule = '待复核', [], '不能据字面规则确定；保留原点，屏蔽该例实体损失。', 'guide:5-9'
                if raw['type'] == 'labels':
                    exact = [e for e in s['entities'] if (e['start'],e['end']) == (raw['start'],raw['end'])]
                    overlaps = [e for e in s['entities'] if type(raw['start']) is int and type(raw['end']) is int and
                                max(e['start'],raw['start']) < min(e['end'],raw['end'])]
                    final = [e['id'] for e in (exact or overlaps)]
                    same = [e for e in exact if e['label'] in raw['labels']]
                    if same and set(e['label'] for e in same) == set(raw['labels']):
                        status, reason = '保留', '同跨度标签在样例中保留；属性独立核对。'
                    elif final:
                        status, reason = '修订', '按最小完整跨度和指标/数值/单位、来源类别分别重建；重叠只作追溯，不证明标签等价。'
                    elif (any(l in schema['controls']['epidemics_labels'] for l in raw['labels']) or
                          any(i['ref']==raw['ref'] and i['code'] in ('label_text_semantic_candidate','invalid_offsets','span_text_mismatch','label_control_mismatch') for i in p['issues'])):
                        status, reason = '删除', '选区不满足当前标签操作定义或结构合同；不推断未写事实。'
                    elif raw['text'] in ('今','今日','昨天','昨日') or re.fullmatch(r'\d+(?:\.\d+)?(?:℃|%|天|月|年|周)?', raw['text']):
                        status, reason = '删除', '孤立时间/数值没有明确已标事件归属；不把数字当症状或凭最近距离连线。'
                    elif TIME.fullmatch(raw['text']) or re.fullmatch(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?',raw['text']):
                        status, reason = '不沿用', '日期不是诊断/病原实体；时间点归属未确定时不保留原重复标签，已明确检验组另建时间。'
                    elif re.search(r'神志清|神清|意识清|意识状态：清醒|呼吸平|胸闷心悸|气平|气促|上呼吸道感染|^呼吸道感染$|产酸克雷伯|鮑曼不动杆菌|^气管插管$|^胸部CT|地方病地区|慢性炎症|两肺纹理|未见明显异常密度影|无隆起或凹陷|无充盈缺损|符合心功能不全',raw['text']):
                        status, reason = '删除', '正常体征、孤立气促、其他病原、气管插管或明确非感染肺部解释不符合所标实体定义（字典3/4/11章）。'
                    referenced.update(final)
                elif raw['type'] == 'choices':
                    field = raw['from_name']
                    rule = 'guide:8' if field != 'case_decision' else 'guide:' + s['case_rule']
                    if field == 'case_decision':
                        final = [s['case_decision']]
                        vals = [schema['fields'][field]['aliases'].get(v,v) for v in raw['choices']]
                        status = '保留' if vals == final else '修订'
                        reason = s['case_reason']
                    else:
                        owners = old_ids.get(raw['region_id'], [])
                        candidates = [e for e in s['entities'] if any((e['start'],e['end'],e['label']) == (o['start'],o['end'],o['label']) for o in owners)]
                        final = [{'entity_id':e['id'],'field':field,'value':e['attributes'][field]} for e in candidates if field in e['attributes']]
                        if final:
                            vals = [schema['fields'][field]['aliases'].get(v,v) for v in raw['choices']]
                            status = '保留' if all([v['value']] == vals for v in final) else '修订'
                            reason = '按当前实体所属否定、时间、主体或独立检验组确定，空白不充当阴性。'
                        else:
                            status, reason = '删除随属', '原实体已重划/删除，或该字段不适用；新实体属性另列新增点，不沿用旧ID。'
                elif raw['type'] == 'relation':
                    final = []
                    for r in s['relations']:
                        em = {e['id']:e for e in s['entities']}
                        ends = [(raw['from_id'],r['from_id']),(raw['to_id'],r['to_id'])]
                        if raw['direction'] == 'left':
                            ends = [(raw['to_id'],r['from_id']),(raw['from_id'],r['to_id'])]
                        if all(any((o['start'],o['end'],o['label']) == (em[new]['start'],em[new]['end'],em[new]['label']) for o in old_ids.get(old,[])) for old,new in ends):
                            final.append(r)
                    status = '保留' if len(final)==1 and raw['labels']==[final[0]['type']] else '删除重建' if final else '不沿用'
                    reason, rule = '原缺类型、错误方向或端点不能直接填类；样例只另建有同组文字依据的关系，未标对仍为未知。', 'guide:9.3'
                if status == '待复核':
                    s['ner_supervised'] = False
                ledger.append({'id':raw['ref'], 'task_id':s['task_id'],'source_annotation_id':ann['id'],
                               'kind':raw['type'],'original':raw,'decision':status,'final':final,
                               'reason':reason,'rule_ref':rule,'reviewer':'Codex_AI_rule_assisted_sample',
                               'expert_signed':False})
        for e in s['entities']:
            if e['id'] not in referenced:
                ledger.append({'id':s['task_id']+'/AI/'+e['id'],'task_id':s['task_id'],'kind':'entity_added',
                               'decision':'新增样例','final':e,'reason':e['reason'],'rule_ref':e['rule_ref'],'expert_signed':False})
            for f,v in e['attributes'].items():
                ledger.append({'id':s['task_id']+'/AI/'+e['id']+'/'+f,'task_id':s['task_id'],'kind':'attribute_final',
                               'decision':'样例定值','final':{'entity_id':e['id'],'field':f,'value':v},
                               'evidence':e['attribute_evidence'],'reason':e['reason'],'rule_ref':'guide:8','expert_signed':False})
        for i,r in enumerate(s['relations']):
            ledger.append({'id':s['task_id']+'/AI/rel'+str(i),'task_id':s['task_id'],'kind':'relation_final',
                           'decision':'新增样例','final':r,'reason':r['reason'],'rule_ref':r['rule_ref'],'expert_signed':False})
    return ledger, {'cases':len(samples),'original_results':raw_count,'point_rows':len(ledger),
                    'decisions':dict(Counter(p['decision'] for p in ledger)),
                    'pending_points':sum(p['decision']=='待复核' for p in ledger),
                    'ner_masked_cases':sum(not s['ner_supervised'] for s in samples),
                    'entities':sum(len(s['entities']) for s in samples),
                    'relations':sum(len(s['relations']) for s in samples),
                    'case_counts':dict(Counter(s['case_decision'] for s in samples)), 'expert_signed_cases':0}


def to_label_studio(samples, schema):
    output = []
    for s in samples:
        results = []
        for e in s['entities']:
            control = next(c for c, labels in schema['controls'].items() if e['label'] in labels)
            value = {k:e[k] for k in ('start','end','text')}
            results.append({'id':e['id'],'from_name':control,'to_name':'chief_complaint_text','type':'labels','value':{**value,'labels':[e['label']]}})
            for f,v in e['attributes'].items():
                results.append({'id':e['id'],'from_name':f,'to_name':'chief_complaint_text','type':'choices','value':{**value,'choices':[v]}})
        for r in s['relations']:
            results.append({'type':'relation','from_id':r['from_id'],'to_id':r['to_id'],'direction':'right','labels':[r['type']]})
        results.append({'type':'choices','from_name':'case_decision','to_name':'chief_complaint_text','value':{'choices':[s['case_decision']]}})
        # Predictions deliberately require confirmation after import, not submitted expert annotations.
        output.append({'data':{'text':s['text'],'task_id':s['task_id'],'reference_status':s['reference_status']},
                       'predictions':[{'model_version':'ai_sample_adjudication_001','result':results}]})
    return output


def build_review_html(samples, ledger, counts):
    payload = json.dumps({'samples':samples,'points':ledger,'counts':counts},ensure_ascii=False).replace('<','\\u003c')
    template = Path(__file__).with_name('templates') / 'ai_sample_review.html'
    return template.read_text('utf-8').replace('__PAYLOAD__',payload)


if __name__ == '__main__':
    main()
