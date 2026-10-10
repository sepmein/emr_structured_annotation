"""Offline, non-destructive audit of Label Studio double annotations.

Every exported result gets an inventory row. Review rows use semantic spans,
never cross-annotator region UUIDs. No medical decisions are auto-applied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from emr_annotation.annotation.schema import load_double_annotation_schema as load_schema
from emr_annotation.adjudication.workbench import build_html


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def safe_text(text):
    # Minimise identifiers in derived excerpts; this is not a deidentification certification.
    text = re.sub(r'((?:姓名|身份证(?:号)?|联系电话|住院号|床号)[ \t]*[:：][ \t]*)[^\n\r，。;；]*', r'\1[已隐藏]', text)
    text = re.sub(r'(?<!\d)\d{11,}(?:[Xx])?(?!\d)', '[长编号已隐藏]', text)
    return text


def context(text, start, end):
    if not isinstance(start, int) or not isinstance(end, int):
        return ''
    return safe_text(text[max(0, start - 38):min(len(text), end + 38)])


def display_windows(text, ranges, padding=45):
    """Minimum necessary excerpts, masked without changing character offsets.

    Windows preserve Python character coordinates. Browser rendering must use
    Array.from(), not UTF-16 String.slice(), to match those coordinates.
    """
    masked = list(text)
    sensitive = [m.span(2) for m in re.finditer(
        r'((?:姓名|身份证(?:号)?|联系电话|住院号|床号)[ \t]*[:：][ \t]*)([^\n\r，。;；]*)', text)]
    sensitive += [m.span() for m in re.finditer(r'(?<!\d)\d{11,}(?:[Xx])?(?!\d)',text)]
    for start,end in sensitive:
        for i in range(start,end):
            if masked[i] not in '\n\r\t ': masked[i]='■'
    spans=[]
    for start,end in ranges:
        if type(start) is int and type(end) is int and 0 <= start < end <= len(text):
            spans.append((max(0,start-padding),min(len(text),end+padding)))
    if not spans and text:
        spans=[(0,min(len(text),260))]
    merged=[]
    for start,end in sorted(spans):
        if merged and start <= merged[-1][1]: merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else: merged.append((start,end))
    return [dict(start=s,end=e,text=''.join(masked[s:e])) for s,e in merged]


def metric(a, b):
    a, b = Counter(a), Counter(b)
    common = sum((a & b).values())
    na, nb = sum(a.values()), sum(b.values())
    return dict(A=na, B=nb, exact=common, A_only=na-common, B_only=nb-common,
                symmetric_f1=round(2*common/(na+nb), 6) if na+nb else None)


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    ca, cb = Counter(a for a, b in pairs), Counter(b for a, b in pairs)
    observed = sum(a == b for a, b in pairs)/n
    expected = sum(ca[k]*cb[k] for k in ca.keys() | cb.keys())/n**2
    return dict(n=n, agree=sum(a == b for a, b in pairs), agreement=observed,
                kappa=(observed-expected)/(1-expected) if expected < 1 else None,
                matrix=[dict(A=a, B=b, count=c) for (a,b),c in sorted(Counter(pairs).items())])


def parse_annotation(task, annotation, schema, alias, task_key):
    text = task['data'].get('text', '')
    entities, attrs, relations, cases, inventory, issues = [], defaultdict(dict), [], [], [], []
    regions = defaultdict(list)
    def issue(code, ref, detail, severity='P1'):
        issues.append(dict(code=code, ref=ref, detail=detail, severity=severity))
    for i, result in enumerate(annotation.get('result', []), 1):
        ref = f'{task_key}/{alias}/R{i:03d}'
        value = result.get('value', {})
        row = dict(ref=ref, type=result.get('type'), region_id=result.get('id'),
                   from_name=result.get('from_name'), to_name=result.get('to_name'),
                   start=value.get('start'), end=value.get('end'),
                   text=safe_text(value.get('text', '')), labels=result.get('labels', value.get('labels', [])),
                   choices=value.get('choices', []), from_id=result.get('from_id'),
                   to_id=result.get('to_id'), direction=result.get('direction'))
        inventory.append(row)
        if row['type'] in ('labels', 'choices') and row['to_name'] != 'chief_complaint_text':
            issue('wrong_text_target', ref, str(row['to_name']))
        if row['type'] == 'labels':
            start, end = row['start'], row['end']
            valid = (type(start) is int and type(end) is int and 0 <= start < end <= len(text))
            if not valid:
                issue('invalid_offsets', ref, f'{start}:{end}', 'P0')
            elif value.get('text') != text[start:end]:
                issue('span_text_mismatch', ref, '导出选区文字与data.text切片不一致', 'P0')
            if value.get('text','') != value.get('text','').strip():
                issue('span_whitespace_candidate',ref,'选区含边缘空格，核对最小完整跨度')
            if not row['labels']:
                issue('empty_entity_label', ref, '实体没有标签')
            if len(row['labels']) > 1:
                issue('multiple_labels_in_control', ref, '同一控件选区有多个标签')
            allowed = schema['controls'].get(row['from_name'], [])
            for label in row['labels']:
                if label not in allowed:
                    issue('label_control_mismatch', ref, f'{row["from_name"]}:{label}')
                ent = dict(ref=ref, id=row['region_id'], start=start, end=end, label=label,
                           text=row['text'], context=context(text,start,end), control=row['from_name'])
                entities.append(ent)
                regions[row['region_id']].append(ent)
                token = value.get('text', '').strip()
                indicators = set(schema['controls'].get('measure_entities', []))
                if ((label in indicators and (re.fullmatch(r'[\d.\-–]+', token) or token in ('℃','%','次/分','度')))
                    or (label == '数值' and not re.fullmatch(r'[\d.\-–~～]+', token))
                    or (label == '单位' and re.fullmatch(r'[\d.]+', token))
                    or (label == '比较符' and token in ('%','℃','次/分'))
                    or (label == '发热' and re.fullmatch(r'[\d一二三四五六七八九十半]+(?:天|周|月|年|小时)(?:余|前)?', token))
                    or (label == '肺炎衣原体' and token == '肺炎支原体')
                    or (label == '三凹征' and token == '意识障碍')):
                    issue('label_text_semantic_candidate', ref, f'{label}与选区用语不符，按字典逐项人工确认')
        elif row['type'] == 'choices':
            field = row['from_name']
            rule = schema['fields'].get(field)
            if not rule:
                issue('unknown_choice_field', ref, str(field)); continue
            choices = [rule['aliases'].get(c, c) for c in row['choices']]
            if len(choices) != 1 or any(c not in rule['values'] for c in choices):
                issue('invalid_choice', ref, f'{field}:{choices}')
            if field == 'case_decision':
                cases.append(dict(ref=ref, values=choices))
                if row['start'] is not None:
                    issue('case_decision_has_span', ref, '病例结论不能绑定选区')
            else:
                rid = row['region_id']
                if field in attrs[rid]:
                    issue('duplicate_attribute', ref, f'{field}重复记录')
                attrs[rid].setdefault(field, []).append(dict(ref=ref, values=choices,
                                                          start=row['start'], end=row['end']))
        elif row['type'] == 'relation':
            # Label Studio relationship labels are top-level, not value.labels.
            rel = dict(ref=ref, from_id=row['from_id'], to_id=row['to_id'],
                       direction=row['direction'], labels=result.get('labels', []))
            relations.append(rel)
        else:
            issue('unknown_result_type', ref, str(row['type']))
    if len(cases) != 1:
        issue('invalid_case_count', f'{task_key}/{alias}', f'病例结论记录数={len(cases)}')
    for rid, ents in regions.items():
        spans = {(e['start'],e['end']) for e in ents}
        if len(spans) != 1:
            issue('region_id_span_collision', ents[0]['ref'], '同一ID对应多个跨度', 'P0')
        if len({e['label'] for e in ents}) > 1:
            issue('multi_label_region', ents[0]['ref'], '同一选区ID标签：'+','.join(e['label'] for e in ents))
        if len(ents) != len({(e['start'],e['end'],e['label']) for e in ents}):
            issue('duplicate_entity', ents[0]['ref'], '同一ID同一标签重复')
        for field, rule in schema['fields'].items():
            if not rule['per_region']: continue
            applicable = any(e['label'] in rule['scope'] for e in ents)
            records = attrs.get(rid, {}).get(field, [])
            if applicable and rule['required'] and not records:
                issue('required_attribute_missing', ents[0]['ref'], field)
            if records and not applicable:
                issue('attribute_not_applicable', records[0]['ref'], field)
            for rec in records:
                if (rec['start'],rec['end']) not in spans:
                    issue('attribute_span_mismatch', rec['ref'], field, 'P0')
        for e in ents:
            e['attributes'] = {f: [v for r in rs for v in r['values']] for f,rs in attrs.get(rid, {}).items()}
            e['attribute_refs'] = {f: [r['ref'] for r in rs] for f,rs in attrs.get(rid, {}).items()}
    for rid, fields in attrs.items():
        if rid not in regions:
            for records in fields.values():
                issue('orphan_attribute', records[0]['ref'], '属性找不到同ID实体', 'P0')
    identical=defaultdict(list)
    for ent in entities:
        identical[entity_key(ent)].append(ent)
        if (ent.get('attributes',{}).get('finding_context') == ['known_absent']
            and ent.get('attributes',{}).get('clinical_course')):
            issue('absent_with_clinical_course',ent['ref'],'明确不存在的实体不填写病程变化（指南§8.1）')
    for ents in identical.values():
        if len(ents)>1 and len({e['id'] for e in ents})>1:
            issue('duplicate_span_label_different_ids',ents[0]['ref'],'相同跨度标签被多个ID重复标注，逐个核对后合并')
    for rel in relations:
        if rel['from_id'] == rel['to_id']:
            issue('self_relation', rel['ref'], '关系连接自身，删除或重建端点', 'P0')
        if rel['from_id'] not in regions or rel['to_id'] not in regions:
            issue('dangling_relation', rel['ref'], '关系端点不存在', 'P0')
        if len(rel['labels']) != 1 or rel['labels'][0] not in schema['relations']:
            issue('relation_type_missing_or_invalid', rel['ref'], '关系类型未保存或不合法', 'P0')
        if rel['direction'] not in ('right', 'left'):
            issue('relation_direction_invalid', rel['ref'], str(rel['direction']), 'P0')
        src, dst = regions.get(rel['from_id'], []), regions.get(rel['to_id'], [])
        if rel['direction'] == 'left': src, dst = dst, src
        rel['source'] = sorted({(e['start'],e['end'],e['label']) for e in src})
        rel['target'] = sorted({(e['start'],e['end'],e['label']) for e in dst})
        rel['context'] = context(text, min((e['start'] for e in src+dst if type(e['start']) is int), default=0),
                                 min(len(text), max((e['end'] for e in src+dst if type(e['end']) is int), default=0)))[:240]
        if src and dst:
            sl,dl={e['label'] for e in src},{e['label'] for e in dst}
            indicators=set(schema['controls'].get('measure_entities',[]))
            components=set(schema['controls'].get('measure_labels',[]))
            possible_measure=bool(sl <= indicators and dl <= components)
            possible_time=bool(dl == {'时间表达'} and not (sl & (components | {'时间表达'})))
            if not possible_measure and not possible_time:
                issue('relation_topology_candidate',rel['ref'],'现有端点类别/方向不能对应五种关系；先修正实体再重连')
        if len(rel['labels']) == 1 and src and dst:
            name = rel['labels'][0]
            sl, dl = {e['label'] for e in src}, {e['label'] for e in dst}
            indicators = set(schema['controls'].get('measure_entities', []))
            components = set(schema['controls'].get('measure_labels', []))
            valid = bool(sl <= indicators and dl <= components) if name == '测量' else (
                dl == {'时间表达'} and not (sl & (components | {'时间表达'})))
            if not valid: issue('relation_endpoint_type_invalid', rel['ref'], name, 'P0')
    return dict(alias=alias, annotator=annotation.get('completed_by'), annotation_id=annotation.get('id'),
                created_at=annotation.get('created_at'), updated_at=annotation.get('updated_at'),
                updated_by=annotation.get('updated_by'), lead_time=annotation.get('lead_time'),
                entities=entities, relations=relations, case=cases, inventory=inventory, issues=issues)


def entity_key(e):
    return (e['start'],e['end'],e['label'])


def relation_key(r, typed=True):
    return (tuple(map(tuple,r['source'])), tuple(map(tuple,r['target'])),
            tuple(r['labels']) if typed else ())


def compare_task(a, b, schema):
    rows = []
    def row(kind, status, av, bv, rule, ctx=''):
        rows.append(dict(kind=kind,status=status,A=av,B=bv,rule=rule,context=ctx))
    ac = a['case'][0]['values'] if len(a['case']) == 1 else []
    bc = b['case'][0]['values'] if len(b['case']) == 1 else []
    row('病例结论', '一致' if len(ac)==len(bc)==1 and ac==bc else '分歧或缺失',ac,bc,'指南§3–4；先核定索引事件，再核对直接肺部证据及B1–B6')
    am, bm = defaultdict(list), defaultdict(list)
    for e in a['entities']: am[entity_key(e)].append(e)
    for e in b['entities']: bm[entity_key(e)].append(e)
    attr_stats = defaultdict(Counter)
    for key in sorted(am.keys() | bm.keys(), key=str):
        ea, eb = am.get(key,[]), bm.get(key,[])
        status = '一致' if len(ea)==len(eb)==1 else '重复或数量不一致' if ea and eb else '仅A' if ea else '仅B'
        ctx = (ea or eb)[0]['context']
        row('实体',status,ea,eb,'字典对应标签卡；指南§5跨度与重复、§6实体。缺标只表示一方未标，不能自动认定另一方正确',ctx)
        if not ea or not eb:
            opposite = b['entities'] if ea else a['entities']
            rows[-1]['overlap_candidates'] = [dict(ref=e['ref'],start=e['start'],end=e['end'],label=e['label'],text=e['text'])
                for e in opposite if type(e['start']) is int and type(key[0]) is int and
                max(e['start'],key[0]) < min(e['end'],key[1])]
        if len(ea)==len(eb)==1:
            for field, rule in schema['fields'].items():
                if not rule['per_region'] or key[2] not in rule['scope']: continue
                va, vb = ea[0]['attributes'].get(field,[]),eb[0]['attributes'].get(field,[])
                if not va and not vb:
                    status = '双方漏填必填' if rule['required'] else '双方空白（选填）'
                elif va == vb: status = '一致'
                else: status = '分歧或单方漏填'
                row('属性',status,dict(entity=list(key),field=field,values=va,ref=ea[0]['ref'],result_refs=ea[0]['attribute_refs'].get(field,[])),
                    dict(entity=list(key),field=field,values=vb,ref=eb[0]['ref'],result_refs=eb[0]['attribute_refs'].get(field,[])),
                    '指南§8；空白、无法判断、原文未说明、不适用分别处理；先核对否定作用域、时间、主体及同组检验',ctx)
                attr_stats[field]['matched_slots']+=1
                if not va and not vb: attr_stats[field]['both_blank']+=1
                elif va==vb: attr_stats[field]['equal_nonempty']+=1
                elif not va or not vb: attr_stats[field]['one_blank']+=1
                else: attr_stats[field]['different_nonempty']+=1
    ar, br = defaultdict(list),defaultdict(list)
    for r in a['relations']: ar[relation_key(r)].append(r)
    for r in b['relations']: br[relation_key(r)].append(r)
    for key in sorted(ar.keys() | br.keys(),key=str):
        ra,rb=ar.get(key,[]),br.get(key,[])
        typed = len(key[2])==1 and key[2][0] in schema['relations']
        status = ('一致' if len(ra)==len(rb)==1 else '数量分歧' if ra and rb else '仅A' if ra else '仅B') if typed else '缺关系类型（必须重核）'
        row('关系',status,ra,rb,'指南§9；确认端点、方向、类型和同组归属。缺类型不能按端点自动回填', (ra or rb)[0]['context'])
    for alias,parsed in [('A',a),('B',b)]:
        for issue in parsed['issues']:
            row('结构问题',issue['severity'],issue if alias=='A' else None,issue if alias=='B' else None,
                '逐条修复后回导出复检；两人同错也须裁决')
    return rows,attr_stats


def dictionary_cards(path, labels):
    content=Path(path).read_text('utf-8')
    sections=re.split(r'(?m)^### ', content)
    cards={}
    for label in labels:
        hits=[s for s in sections if re.match(r'\d+\.\d+ '+re.escape(label)+r'\s*(?:\n|$)',s)]
        if hits:
            s=hits[0]
            cards[label]=dict(section=s.splitlines()[0],
                checks=[line[2:] for line in s.splitlines() if line.startswith('- **') and
                        any(k in line for k in ('项目操作定义','纳入标准','排除标准','最小完整跨度','可填属性','允许关系','常见混淆'))])
        else: cards[label]=dict(section='需查字典对应条款',checks=[])
    return cards


def run(inputs, configs, output, guide, dictionary, review_notes=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空；请用新的运行目录，保留此前裁决记录')
    schemas={p:load_schema(c) for p,c in configs.items()}
    schema=next(iter(schemas.values()))
    cards=dictionary_cards(dictionary,schema['labels'])
    notes=json.loads(Path(review_notes).read_text('utf-8')) if review_notes else {}
    tasks=[]; inventory=[]; all_rows=[]; project_metrics={}; sources=[]
    all_label_a,all_label_b=[],[]
    label_stats=defaultdict(lambda:Counter(A=0,B=0,exact=0))
    field_stats=defaultdict(Counter)
    pairs_all=[]
    for path in inputs:
        path=Path(path); data=json.loads(path.read_text('utf-8'))
        if not isinstance(data,list): raise ValueError(f'{path.name}:需要Label Studio任务数组')
        projects={str(t.get('project')) for t in data}
        if len(projects)!=1: raise ValueError('每个输入需仅包含一个项目')
        project=next(iter(projects))
        if project not in schemas: raise ValueError(f'项目{project}缺少明确的配置绑定')
        active_schema=schemas[project]
        if active_schema['labels']!=schema['labels']: raise ValueError('项目标签集合不同，请分开审计')
        users=sorted({str(a.get('completed_by')) for t in data for a in t.get('annotations',[]) if not a.get('was_cancelled')})
        if len(users)!=2: raise ValueError(f'项目{project}:有效completed_by不是两人，请先明确版本/人员选择')
        alias_by_user={users[0]:'A',users[1]:'B'}
        pairs=[]; entity_a=[]; entity_b=[]; typed_a=[]; typed_b=[]; endpoint_a=[];endpoint_b=[]
        seen=set()
        for task in data:
            ls_id=task['id']; business_id=str(task['data'].get('task_id',''))
            key=f'P{project}/LS{ls_id}/{business_id}'
            if ls_id in seen: raise ValueError('项目内重复任务ID，不能静默去重')
            seen.add(ls_id)
            anns=[a for a in task.get('annotations',[]) if not a.get('was_cancelled')]
            counts=Counter(str(a.get('completed_by')) for a in anns)
            if len(anns)!=2 or set(counts)!=set(users) or any(v!=1 for v in counts.values()):
                raise ValueError(f'{key}:缺失双标或同人多版本，不能自动选最新版本')
            pars={alias_by_user[str(a.get('completed_by'))]:parse_annotation(task,a,active_schema,alias_by_user[str(a.get('completed_by'))],key) for a in anns}
            a,b=pars['A'],pars['B']
            rows,ats=compare_task(a,b,active_schema)
            note=notes.get(f'{project}/{business_id}')
            if note:
                rows.append(dict(kind='病例复核建议',status='待人工确认',A=a['case'],B=b['case'],
                    rule=note['rule'],context=note['recommendation'],evidence_offsets=note.get('evidence_offsets',[])))
            for f,s in ats.items(): field_stats[f].update(s)
            for i,row in enumerate(rows,1):
                row.update(id=f'{key}/Q{i:04d}',task=key,project=project,business_task_id=business_id)
            all_rows.extend(rows)
            for side in ('A','B'):
                for r in pars[side]['inventory']: inventory.append(dict(task=key,project=project,side=side,**r))
            ca=a['case'][0]['values'] if len(a['case'])==1 else []
            cb=b['case'][0]['values'] if len(b['case'])==1 else []
            if len(ca)==len(cb)==1 and ca[0] in active_schema['fields']['case_decision']['values'] and cb[0] in active_schema['fields']['case_decision']['values']:
                pairs.append((ca[0],cb[0]));pairs_all.append((ca[0],cb[0]))
            ka,kb=[entity_key(e) for e in a['entities']],[entity_key(e) for e in b['entities']]
            entity_a.extend((key,*k) for k in ka);entity_b.extend((key,*k) for k in kb)
            for label in schema['labels']:
                m=metric([k for k in ka if k[2]==label],[k for k in kb if k[2]==label])
                for f in ('A','B','exact'): label_stats[label][f]+=m[f]
            for side,typed,endpoints in [('A',typed_a,endpoint_a),('B',typed_b,endpoint_b)]:
                for r in pars[side]['relations']:
                    endpoints.append((key,relation_key(r,False)))
                    if len(r['labels'])==1 and r['labels'][0] in active_schema['relations'] and r['source'] and r['target']:
                        typed.append((key,relation_key(r)))
            text=task['data'].get('text','')
            privacy_candidates=[]
            for m in re.finditer(r'(姓名|身份证(?:号)?|联系电话|住院号|床号)[ \t]*[:：][ \t]*([^\n\r，。;；]*)',text):
                value=m.group(2).strip()
                if value and not re.fullmatch(r'\[[^\]]*\]|[_*Xx某/\s]+',value):
                    privacy_candidates.append(dict(field=m.group(1),start=m.start(),end=m.end()))
            tasks.append(dict(key=key,project=project,ls_task_id=ls_id,business_task_id=business_id,
                text_length=len(text),text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                metadata_identifier_fields=[f for f in ('patient_id','serial_number','patient_group_id') if task['data'].get(f)],
                privacy_candidates=privacy_candidates, annotations=pars,review_rows=rows,
                display_windows=display_windows(text,[(e['start'],e['end']) for side in pars.values() for e in side['entities']]
                    + [tuple(span) for span in (note or {}).get('evidence_offsets',[])]),
                entity_metric=metric(ka,kb), case_A=ca,case_B=cb,
                fully_equal=all(r['status'] in ('一致','双方空白（选填）') for r in rows if r['kind']!='病例复核建议')))
        project_metrics[project]=dict(tasks=len(data),annotators=alias_by_user,case=kappa(pairs),
            entity=metric(entity_a,entity_b),typed_relation=metric(typed_a,typed_b),
            relation_endpoints_only=metric(endpoint_a,endpoint_b))
        all_label_a.extend(entity_a);all_label_b.extend(entity_b)
        sources.append(dict(file=path.name,sha256=digest(path),project=project,
                            schema_file=str(configs[project]),schema_sha256=digest(configs[project])))
    text_groups=defaultdict(list)
    for t in tasks: text_groups[t['text_sha256']].append(t['key'])
    duplicate_groups=[v for v in text_groups.values() if len(v)>1]
    issue_counts=Counter(i['code'] for t in tasks for p in t['annotations'].values() for i in p['issues'])
    stats=[]
    for label in schema['labels']:
        s=label_stats[label]; denominator=s['A']+s['B']
        stats.append(dict(label=label,**s,A_only=s['A']-s['exact'],B_only=s['B']-s['exact'],
            f1=round(2*s['exact']/denominator,6) if denominator else None,
            coverage='出现' if denominator else '本批未出现，不能评价',card=cards[label]))
    summary=dict(tasks=len(tasks),annotations=sum(len(t['annotations']) for t in tasks),
        result_rows=len(inventory),review_rows=len(all_rows),
        pending_rows=sum(r['status'] not in ('一致','双方空白（选填）') for r in all_rows),
        fully_equal_tasks=sum(t['fully_equal'] for t in tasks),case=kappa(pairs_all),
        entities=metric(all_label_a,all_label_b),projects=project_metrics,issues=dict(issue_counts),
        duplicate_text_groups=duplicate_groups,privacy_candidates=sum(bool(t['privacy_candidates']) for t in tasks))
    review_context=dict(guide_sha256=digest(guide),dictionary_sha256=digest(dictionary),
                        notes_sha256=digest(review_notes) if review_notes else None)
    payload=dict(version='1.0',sources=sources,review_context=review_context,guide=dict(file=str(guide),sha256=digest(guide)),
        dictionary=dict(file=str(dictionary),sha256=digest(dictionary)),schema=schema,
        summary=summary,labels=stats,fields={f:dict(field_stats[f]) for f in schema['fields'] if f!='case_decision'},
        tasks=tasks,inventory=inventory,review_rows=all_rows,
        limitations=['导出不能证明双标盲法独立；updated_by可能与completed_by不同',
          '现有XML作为候选核查合同；实际项目当时配置需负责人核实并保存哈希',
          '严格实体F1为原始导出跨度+标签的对称一致性，不是医学准确率；错误结果仍保留在描述性统计中',
          '不同项目的A/B是不同人员，合并kappa只作描述，正式解释以各项目结果为主',
          '未发现关键词不代表隐私合格；元数据患者/就诊标识须确认已去标识化',
          '属性只在单一精确匹配实体上比较；未匹配实体的全部属性在实体行保留',
          '关系类型缺失不推断；端点一致性不能代替严格关系F1',
          '两人一致也可能同错；本报告尚未由医学裁决员逐条签署'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
    (output/'review.html').write_text(build_html(payload),'utf-8')
    guide_path = Path(__file__).resolve().parents[2] / 'documentation/double_annotation_adjudication/double-annotation-adjudication.md'
    try:
        guide_link = Path(os.path.relpath(guide_path, output.resolve())).as_posix()
    except ValueError:
        # Windows outputs can live on a different drive from the repository.
        guide_link = guide_path.as_uri()
    (output/'核查报告.md').write_text(build_report(payload, guide_link),'utf-8')
    (output/'裁决模板.json').write_text(json.dumps(dict(version='1.0',sources=sources,review_context=review_context,
        decisions=[dict(id=r['id'],task=r['task'],kind=r['kind'],status='待裁决',final_answer='',
                        evidence_offsets=[],rule_ref='',reason_category='',reason='',adjudicator='',
                        reviewed_at='',needs_back_annotation=False) for r in all_rows]),ensure_ascii=False,indent=2),'utf-8')
    (output/'标签核查签署模板.json').write_text(json.dumps(dict(sources=sources,review_context=review_context,
        labels=[dict(label=l['label'],A_count=l['A'],B_count=l['B'],coverage=l['coverage'],status='待核查',
                     reviewed_tasks=[],new_missing_mentions=[],review_result='',rule_ref=l['card']['section'],
                     adjudicator='',reviewed_at='') for l in stats]),ensure_ascii=False,indent=2),'utf-8')
    return summary


def build_report(p, guide_link='../../../../documentation/double_annotation_adjudication/double-annotation-adjudication.md'):
    s=p['summary']
    lines=['# 双人标注核查报告','',
        '本报告检查原始导出，未替任何一方作医学裁决。输入不覆盖；后续批次用新目录运行。',
        f"覆盖：{s['tasks']}例、{s['annotations']}份标注、{s['result_rows']}条原始结果；生成{s['review_rows']}条核查项，其中{s['pending_rows']}条需要处理。",
        '', '## 项目内一致性', '', '| 项目 | 双标任务 | 病例一致 | kappa | 原始实体严格F1 | 关系类型 |', '|---|---:|---:|---:|---:|---|']
    for project,m in s['projects'].items():
        k=m['case'];f=m['entity']['symmetric_f1']
        case_text=f"{k['agree']}/{k['n']}" if k is not None else '不可评价（无可比病例）'
        kappa_text=f"{k['kappa']:.4f}" if k is not None and k['kappa'] is not None else '不可评价'
        f1_text=f"{f:.4f}" if f is not None else '不可评价'
        lines.append(f"| {project} | {m['tasks']} | {case_text} | {kappa_text} | {f1_text} | 合法类型A={m['typed_relation']['A']}，B={m['typed_relation']['B']} |")
    overall_case=s['case']
    overall_case_text=f"{overall_case['agree']}/{overall_case['n']}" if overall_case is not None else '不可评价（无可比病例）'
    lines.extend(['','病例总体一致：'+overall_case_text+'；不同项目A/B不是同一对人，合并指标仅作描述。',
        '实体一致性按病例内(start,end,label)多重集计算：2×精确共同项/(A实体数+B实体数)，不把UUID或控件名当医学标签。重叠跨度只供候选核对，不算精确一致。',
        '原始错误也留在描述性统计；正式验收先修复阻塞问题，再独立复标计算一致性。空集F1记不可评价。关系必须同时具备合法类型、归一化端点和方向；无类型关系另列，不能评分为正确关系。',
        '', '## 结构核查', '', '| 问题代码 | 条数 |', '|---|---:|'])
    for code,n in sorted(s['issues'].items()):lines.append(f'| {code} | {n} |')
    lines.extend(['','## 逐病例结果','','| 项目/业务任务 | LS任务 | A结论 | B结论 | A/B实体 | 精确共同实体 | 待处理项 |','|---|---:|---|---|---:|---:|---:|'])
    for t in p['tasks']:
        m=t['entity_metric'];pending=sum(r['status'] not in ('一致','双方空白（选填）') for r in t['review_rows'])
        lines.append(f"| {t['project']}/{t['business_task_id']} | {t['ls_task_id']} | {'/'.join(t['case_A'])} | {'/'.join(t['case_B'])} | {m['A']}/{m['B']} | {m['exact']} | {pending} |")
    lines.extend(['','## 逐病例人工复核建议','','这些建议依据项目指南，尚未签署；“建议保留”也不是已完成医学复核。',
                  '', '| 病例 | 核查建议 | 规则 |', '|---|---|---|'])
    for r in p['review_rows']:
        if r['kind']=='病例复核建议':lines.append(f"| {r['project']}/{r['business_task_id']} | {r['context'].replace('|','/')} | {r['rule']} |")
    lines.extend(['','## 49个实体标签逐项核查','','0/0表示未覆盖，不能称该标签一致率100%。标签定义及可操作检查点已从当前字典提取到页面各标签卡。',
        '', '| 标签 | A | B | 精确共同 | 仅A | 仅B | 严格F1 |', '|---|---:|---:|---:|---:|---:|---:|'])
    for l in p['labels']:lines.append(f"| {l['label']} | {l['A']} | {l['B']} | {l['exact']} | {l['A_only']} | {l['B_only']} | {l['f1'] if l['f1'] is not None else '未覆盖'} |")
    lines.extend(['','## 9个属性字段逐项核查','','仅在两人都有且各自只有一个相同跨度、相同标签的实体上比较。双方漏填不算已填一致；单方实体遗漏的属性在实体核查项中一并审阅。',
        '', '| 属性 | 可比槽位 | 双方已填相同 | 双方空白 | 单方空白 | 双方已填不同 |', '|---|---:|---:|---:|---:|---:|'])
    for f,c in p['fields'].items():lines.append(f"| {f} | {c.get('matched_slots',0)} | {c.get('equal_nonempty',0)} | {c.get('both_blank',0)} | {c.get('one_blank',0)} | {c.get('different_nonempty',0)} |")
    lines.extend(['','## 解释边界','']+['- '+v for v in p['limitations']])
    lines.extend(['','## 阅读与执行','',
        '打开review.html，先选“需处理”，按项目/病例/类型筛选；每行显示A/B原始答案、偏移、区域ID、最小上下文和规则。再切换“全部”核查双方一致项。',
        'audit.json保留每条原始result的追溯索引，核查项与原始条数不是一一对应：一个实体会拆出多个属性核查项。裁决模板.json涵盖所有核查项。',
        '页面填写状态、最终答案、证据偏移、规则、原因、裁决人和回标要求后，点“导出裁决JSON”。浏览器本地缓存只作临时草稿；定期导出，正式记录使用导出的JSON。',
        '页面含医疗片段，仅在获授权的本地环境使用。输出隐藏直接患者/就诊元数据值；这不等于完成隐私验收。',
        '标签核查签署模板.json提供49个标签的阅读全文查漏签署项，包含未覆盖标签；不是只核对两方已有实体的并集。',
        f'具体分工、顺序、验收与跨批次规则见[可复用裁决方案]({guide_link})。'])
    return '\n'.join(lines)+'\n'





def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--project-config',action='append',required=True,help='PROJECT=XML_PATH；每个项目明确绑定')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--guide',type=Path,default=Path('documentation/annotation/annotation_guide_v2.1.1.md'))
    parser.add_argument('--dictionary',type=Path,default=Path('documentation/annotation/label_dictionary_v2.1.1.md'))
    parser.add_argument('--review-notes',type=Path,help='逐病例人工审阅建议JSON；保留为待确认项')
    args=parser.parse_args()
    configs={k:Path(v) for k,v in (item.split('=',1) for item in args.project_config)}
    print(json.dumps(run(args.input,configs,args.output,args.guide,args.dictionary,args.review_notes),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
