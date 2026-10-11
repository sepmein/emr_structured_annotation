"""Inspectible structured evidence features; these groups are not diagnoses.

No model imports. Missing test results are unknown, never negative. Numerical
values are read only through an explicit measurement edge, never proximity.
"""
import re

SYNDROME_NAMES = ['发热表现', '肺炎证据', '呼吸受损表现', '严重全身表现']
PATHOGEN_NAMES = ['病毒检出', '细菌检出', '真菌检出', '支原体或衣原体检出']
STATES = ['current_present', 'current_absent', 'possible', 'historical',
          'future_or_hypothetical', 'nonpatient', 'unknown', 'resolved']

# These are text cues for the learned case head, not additional Schema entities
# or deterministic case decisions. Meanings come from project guide 3.1-3.2.
CASE_CUE_NAMES=['respiratory_mention','respiratory_negated','respiratory_historical',
                'auscultation_abnormal','uncertain_lung_chest_change','chronic_lung_context',
                'alternative_noninfectious_explanation','explicit_possible_pneumonia',
                'current_worsening_language','same_admission_improvement','unresolved_cross_reference',
                'clinical_narrative_sections','test_list_markers','text_length_scaled']


def project_case_cues(text):
    """Guide-derived observable wording; none is a clinical truth assertion."""
    respiratory=list(re.finditer(r'咳嗽|咳痰|气促|呼吸困难|胸闷|低氧',text))
    negated=historical=0
    for match in respiratory:
        boundary=max(text.rfind(c,0,match.start()) for c in ('。','；','，','\n'))
        prefix=text[boundary+1:match.start()]
        negated+=bool(re.search(r'无|否认|不伴|未诉|未见',prefix))
        historical+=bool(re.search(r'既往|年前|长期|反复.*年',prefix))
    counts=[len(respiratory),negated,historical]
    patterns=[r'湿啰音|肺部啰音',r'脓胸|胸膜瘘|胸腔积液|斑片影|胸部异常',
              r'支气管扩张|间质性肺病|慢性肺|纤维条索|慢性炎症',
              r'心衰|心功能不全|肿瘤|非感染性',r'疑似肺炎|考虑肺炎|不除外肺炎|待排肺炎',
              r'新发|本次.*加重|近日.*加重|急性加重',r'入院.*肺炎|住院.*肺炎|治疗后.*改善|复查.*改善',
              r'病史同前|前症复诊|检查如前|关键附件未提供|正文截断',
              r'主诉|现病史|出院诊断|入院诊断|病程记录',r'检验结果|检测结果|实验室|抗体|IgM|IgG']
    return [min(v/5.,1.) for v in counts]+[min(len(re.findall(p,text))/3.,1.) for p in patterns]+[min(len(text)/2000.,2.)]


def pathogen_groups(schema):
    c = schema['controls']
    return [set(c['pathogen']), set(c['bacteria']), {'曲霉菌', '隐球菌'},
            {'肺炎支原体', '肺炎衣原体', '鹦鹉热衣原体'}]


def clinical_state(entity):
    a = entity.get('attributes', {})
    if a.get('experiencer') in ('family', 'other'):
        return 'nonpatient'
    if a.get('hypothetical') == 'yes' or a.get('conditional') == 'yes' or a.get('temporality') == 'future':
        return 'future_or_hypothetical'
    if a.get('temporality') == 'historical':
        return 'historical'
    if a.get('finding_context') == 'known_absent':
        return 'current_absent'
    if a.get('finding_context') == 'possible':
        return 'possible'
    if a.get('clinical_course') == 'resolved':
        return 'resolved'
    if a.get('finding_context') == 'known_present' and a.get('experiencer') == 'patient' and a.get('temporality') == 'current':
        return 'current_present'
    return 'unknown'


def contract(schema):
    names = [f'{label}:{state}' for label in schema['labels'] for state in STATES]
    pathogens = sorted(set().union(*pathogen_groups(schema)))
    names += [f'{label}:test_{state}' for label in pathogens for state in ('positive','negative','unknown')]
    names += [f'relation:{r}' for r in schema['relations']]
    for label in schema['controls']['measure_entities']:
        names += [f'measurement:{label}:available', f'measurement:{label}:scaled_value']
    names += ['entity_count', 'relation_count', 'explicit_information_gap']
    return dict(version='structured_evidence_v1', features=names, syndrome_names=SYNDROME_NAMES,
                pathogen_names=PATHOGEN_NAMES, syndrome_reference='Derived annotation evidence groups, not independently adjudicated diagnoses',
                pathogen_reference='Detection groups on known result slots; unobserved groups masked',
                numeric_reference='Only explicit measurement relations; exclude explicit historical, planned or other-person contexts; unmarked time is unspecified')


def measurement_context_supported(text, entity):
    """Measurement controls have no required symptom assertion attributes.

    Keep explicitly linked values without inventing current/patient attributes;
    reject explicit incompatible attributes or local clause context.
    """
    if clinical_state(entity) in ('historical', 'future_or_hypothetical', 'nonpatient', 'current_absent'):
        return False
    start = entity['start']
    boundary = max(text.rfind(mark, 0, start) for mark in ('。', '；', '\n'))
    prefix = text[boundary + 1:start]
    return not bool(re.search(r'既往|家属|拟|计划|假设', prefix))


def extract(text, entities, relations, schema):
    spec = contract(schema)
    values = dict.fromkeys(spec['features'], 0.)
    ids = {e['id']: e for e in entities}
    active = set()
    for e in entities:
        state = clinical_state(e)
        key = e['label']+':'+state
        if key in values:
            values[key] = min(1., values[key]+.25)
        if state == 'current_present':
            active.add(e['label'])
        result = e.get('attributes', {}).get('test_result', 'unknown')
        key = e['label']+':test_'+result
        if key in values:
            values[key] = min(1., values[key]+.25)
    for r in relations:
        if r.get('from_id') not in ids or r.get('to_id') not in ids:
            continue
        key = 'relation:'+r['type']
        if key in values:
            values[key] = min(1.,values[key]+.1)
        source, target = ids[r['from_id']], ids[r['to_id']]
        if r['type'] == '测量' and source['label'] in schema['controls']['measure_entities'] and target['label']=='数值' and measurement_context_supported(text, source):
            raw = target.get('text', text[target['start']:target['end']])
            match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*', raw)
            if match:
                number = float(match[1])
                scale = {'体温':50.,'呼吸频率':100.,'静息状态指氧饱和度':100.,
                         '动脉氧分压(PaO2)':150.,'吸氧浓度(FiO2)':100.}[source['label']]
                values[f'measurement:{source["label"]}:available'] = 1.
                values[f'measurement:{source["label"]}:scaled_value'] = max(values[f'measurement:{source["label"]}:scaled_value'], min(number/scale,2.))
    values['entity_count'] = min(len(entities)/50.,2.)
    values['relation_count'] = min(len(relations)/50.,2.)
    values['explicit_information_gap'] = float(bool(re.search(r'关键附件未提供|病史同前|检查如前|信息缺失|正文截断',text)))
    syndromes = [int('发热' in active), int(bool(active & {'肺炎诊断','肺炎影像'})),
                 int(bool(active & {'呼吸衰竭','鼻翼煽动','三凹征','喘鸣喘息','机械通气'})),
                 int(bool(active & {'意识障碍','惊厥','休克','器官衰竭','脱水征'}))]
    pathogen_values, pathogen_masks = [], []
    for group in pathogen_groups(schema):
        observed = [e.get('attributes',{}).get('test_result') for e in entities if e['label'] in group]
        known = [v for v in observed if v in ('positive','negative')]
        pathogen_values.append(int('positive' in known))
        pathogen_masks.append(int(bool(known)))
    return dict(vector=[values[k] for k in spec['features']],syndromes=syndromes,
                pathogens=pathogen_values,pathogen_mask=pathogen_masks,
                active_labels=sorted(active),numeric_evidence={k:v for k,v in values.items() if k.startswith('measurement:') and v})
