"""Integrate export snapshots, describe agreement, and prepare human review.

No answers are selected across conflicting annotation versions. No clinical
adjudication or Label Studio write-back is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from .double_annotations import (compare_task, dictionary_cards, digest,
    display_windows, entity_key, kappa, load_schema, metric, parse_annotation,
    relation_key, build_report)
from emr_annotation.adjudication.workbench import build_html

ROOT = Path(__file__).resolve().parents[2]
CONSISTENT = {'一致', '双方空白（选填）'}
FIELD_NAMES = {'finding_context': '存在状态', 'temporality': '时间属性',
    'clinical_course': '病程变化', 'conditional': '条件性', 'hypothetical': '假设性',
    'experiencer': '主体', 'test_result': '检测结果', 'specimen_type': '标本类型',
    'detection_method': '检测方式'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def submitted_signature(task):
    """Drafts and export metadata are not submitted annotation content."""
    return canonical({'data': task.get('data'), 'annotations': sorted([
        {k: a.get(k) for k in ('id', 'completed_by', 'was_cancelled', 'result')}
        for a in task.get('annotations', [])], key=lambda a: canonical(a.get('id')))})


def integrate(inputs):
    versions = defaultdict(list)
    sources = []
    for path in sorted(map(Path, inputs)):
        data = json.loads(path.read_text('utf-8-sig'))
        if not isinstance(data, list) or not data:
            raise ValueError(f'{path.name}: expected nonempty task array')
        seen = set()
        projects = {str(t.get('project')) for t in data}
        if len(projects) != 1:
            raise ValueError(f'{path.name}: expected one project')
        sources.append({'file': path.name, 'path': str(path.resolve()),
                        'sha256': digest(path), 'project': next(iter(projects)), 'task_rows': len(data)})
        for task in data:
            key = (str(task['project']), task['id'])
            if key in seen:
                raise ValueError(f'{path.name}: duplicate project/task within one export')
            seen.add(key)
            versions[key].append((path, task))
    selected, lineage, blocked = [], [], []
    for (project, task_id), candidates in sorted(versions.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        signatures = {submitted_signature(t) for _, t in candidates}
        metadata_diff = sorted({k for _, t in candidates for k in t
            if any(t.get(k) != other.get(k) for _, other in candidates)})
        # Filename order chooses a container only after submitted content equality.
        path, task = max(candidates, key=lambda v: v[0].name)
        record = {'project': project, 'ls_task_id': task_id,
                  'source_files': [p.name for p, _ in candidates], 'selected_file': path.name,
                  'version_count': len(candidates), 'metadata_difference_keys': metadata_diff,
                  'submitted_signature_sha256': hashlib.sha256(submitted_signature(task).encode()).hexdigest()}
        if len(signatures) > 1:
            record['status'] = 'blocked_conflicting_submitted_content'
            blocked.append(record); lineage.append(record); continue
        anns = [a for a in task.get('annotations', []) if not a.get('was_cancelled')]
        users = [str(a.get('completed_by')) for a in anns]
        if len(anns) != 2 or len(set(users)) != 2 or any(u in ('None', '') for u in users):
            record['status'] = 'blocked_missing_or_ambiguous_pair'
            blocked.append(record); lineage.append(record); continue
        if not isinstance(task.get('data', {}).get('text'), str):
            record['status'] = 'blocked_missing_text'
            blocked.append(record); lineage.append(record); continue
        record['status'] = 'included_identical_submission' if len(candidates) > 1 else 'included'
        record['annotator_pair'] = sorted(users)
        record['annotation_ids'] = [a['id'] for a in anns]
        lineage.append(record); selected.append(task)
    return selected, sources, lineage, blocked


def attribute_statistics(counter, pairs=()):
    slots = counter['matched_slots']
    nonblank = slots - counter['both_blank']
    agreement = kappa(list(pairs))
    if agreement and agreement['n'] < 2:
        agreement = {**agreement, 'kappa': None}
    return {**{k: counter[k] for k in ('matched_slots', 'equal_nonempty', 'both_blank',
                                     'one_blank', 'different_nonempty')},
            'nonblank_slots': nonblank,
            'nonblank_agreement': counter['equal_nonempty'] / nonblank if nonblank else None,
            'blank_inclusive_agreement': (counter['equal_nonempty'] + counter['both_blank']) / slots if slots else None,
            'nonempty_equal_over_all_slots': counter['equal_nonempty'] / slots if slots else None,
            'both_nonempty_kappa': agreement}


def new_accumulator():
    return {'tasks': 0, 'results': 0, 'case_pairs': [], 'entities': [[], []],
            'endpoints': [[], []], 'typed': [[], []], 'invalid_typed': 0,
            'labels': defaultdict(Counter), 'fields': defaultdict(Counter),
            'attribute_pairs': defaultdict(list), 'issues': Counter(), 'result_types': Counter(),
            'typed_by_type': defaultdict(lambda: [[], []]), 'field_coverage': defaultdict(Counter)}


def accumulator_metrics(acc, annotators=None, allow_kappa=True):
    case = kappa(acc['case_pairs'])
    if case and (not allow_kappa or case['n'] < 2):
        case = {**case, 'kappa': None}
    relations = sum(len(v) for v in acc['endpoints'])
    return {'tasks': acc['tasks'], 'result_rows': acc['results'], 'annotators': annotators or {},
            'case': case, 'entity': metric(*acc['entities']),
            'typed_relation': metric(*acc['typed']) if not acc['invalid_typed'] else None,
            'relation_endpoints_only': metric(*acc['endpoints']), 'relation_count': relations,
            'relations_not_fully_typed': acc['invalid_typed'], 'issues': dict(acc['issues']),
            'result_types': dict(acc['result_types'])}


def analyse(tasks, schemas, cards):
    overall = new_accumulator(); projects = {}; groups = {}
    output_tasks, inventory, all_rows = [], [], []
    task_rows, annotation_rows, entity_rows, attribute_rows, relation_rows = [], [], [], [], []
    problem_rows = []
    choice_counts = Counter()
    for raw in tasks:
        project = str(raw['project']); schema = schemas[project]
        anns = [a for a in raw['annotations'] if not a.get('was_cancelled')]
        users = sorted(str(a['completed_by']) for a in anns)
        pair_key = project + ':' + '/'.join(users)
        accs = [overall, projects.setdefault(project, new_accumulator()), groups.setdefault(pair_key, new_accumulator())]
        key = f"P{project}/LS{raw['id']}/{raw['data'].get('task_id', '')}"
        parsed = {side: parse_annotation(raw, next(a for a in anns if str(a['completed_by']) == user), schema, side, key)
                  for side, user in zip(('A', 'B'), users)}
        a, b = parsed['A'], parsed['B']
        rows, attrs = compare_task(a, b, schema)
        ca = a['case'][0]['values'] if len(a['case']) == 1 else []
        cb = b['case'][0]['values'] if len(b['case']) == 1 else []
        # Every case, including two equal answers, gets a human reading checkpoint.
        rows.append({'kind': '病例复核建议', 'status': '待人工确认', 'A': a['case'], 'B': b['case'],
                     'rule': '指南§3–4；阅读全文确认索引事件、肺部证据、否定和时间，检查双方同错',
                     'context': '本例尚未由医学裁决员签署。完成实体、属性和关系检查后再复核病例结论。',
                     'evidence_offsets': []})
        if '1' in users:
            rows.append({'kind': '结构问题', 'status': 'P0',
                'A': {'code': 'annotator_role_unverified', 'detail': '1号账号角色待核实，不能改名为另一标注员或证明独立双标'}, 'B': None,
                'rule': '数据负责人核对人员角色及编辑日志；保留原始身份', 'context': ''})
        for i, r in enumerate(rows, 1):
            r.update(id=f'{key}/Q{i:04d}', task=key, project=project,
                     business_task_id=str(raw['data'].get('task_id', '')), pair_group=pair_key)
        all_rows.extend(rows)
        for acc in accs:
            acc['tasks'] += 1
            if len(ca) == len(cb) == 1 and ca[0] in schema['fields']['case_decision']['values'] and cb[0] in schema['fields']['case_decision']['values']:
                acc['case_pairs'].append((ca[0], cb[0]))
            for f, stats in attrs.items(): acc['fields'][f].update(stats)
        for r in rows:
            if r['kind'] == '属性':
                attribute_rows.append({'task': key, 'project': project, 'pair_group': pair_key,
                    'review_id': r['id'], 'field': r['A']['field'], 'entity': r['A']['entity'],
                    'A': r['A']['values'], 'B': r['B']['values'], 'status': r['status']})
                if (len(r['A']['values']) == len(r['B']['values']) == 1
                    and all(v in schema['fields'][r['A']['field']]['values'] for v in r['A']['values']+r['B']['values'])):
                    for acc in accs: acc['attribute_pairs'][r['A']['field']].append((r['A']['values'][0], r['B']['values'][0]))
        ka, kb = [entity_key(e) for e in a['entities']], [entity_key(e) for e in b['entities']]
        for label in schema['labels']:
            m = metric([k for k in ka if k[2] == label], [k for k in kb if k[2] == label])
            for acc in accs: acc['labels'][label].update({f: m[f] for f in ('A', 'B', 'exact')})
        for side_index, side in enumerate(('A', 'B')):
            p = parsed[side]; ann = next(a for a in anns if str(a['completed_by']) == users[side_index])
            annotation_rows.append({'task': key, 'project': project, 'side': side,
                'annotator': users[side_index], 'annotation_id': ann['id'], 'updated_by': ann.get('updated_by'),
                'updated_by_other': ann.get('updated_by') not in (None, ann.get('completed_by')),
                'created_at': ann.get('created_at'), 'updated_at': ann.get('updated_at'), 'result_rows': len(ann.get('result', []))})
            for r in p['inventory']:
                inventory.append({'task': key, 'project': project, 'side': side, **r})
                for acc in accs: acc['result_types'][r['type']] += 1
                if r['type'] == 'choices':
                    values = r['choices']
                    rule = schema['fields'].get(r['from_name'], {})
                    normalized = [rule.get('aliases', {}).get(v, v) for v in values]
                    choice_counts[(project, pair_key, side, r['from_name'], canonical(normalized))] += 1
            for acc in accs:
                acc['results'] += len(p['inventory'])
                acc['entities'][side_index].extend((key, *entity_key(e)) for e in p['entities'])
                acc['issues'].update(i['code'] for i in p['issues'])
            for e in p['entities']:
                for f, rule in schema['fields'].items():
                    if rule['per_region'] and e['label'] in rule['scope']:
                        for acc in accs:
                            acc['field_coverage'][f][side+'_applicable'] += 1
                            if e['attributes'].get(f): acc['field_coverage'][f][side+'_filled'] += 1
                entity_rows.append({'task': key, 'project': project, 'side': side, 'annotator': p['annotator'],
                    'ref': e['ref'], 'region_id': e['id'], 'start': e['start'], 'end': e['end'],
                    'label': e['label'], 'text': e['text'], 'control': e['control'], 'attributes': e['attributes']})
            for r in p['relations']:
                good = len(r['labels']) == 1 and r['labels'][0] in schema['relations'] and bool(r['source']) and bool(r['target'])
                for acc in accs:
                    acc['endpoints'][side_index].append((key, relation_key(r, False)))
                    if good:
                        acc['typed'][side_index].append((key, relation_key(r)))
                        acc['typed_by_type'][r['labels'][0]][side_index].append((key, relation_key(r)))
                    else: acc['invalid_typed'] += 1
                relation_rows.append({'task': key, 'project': project, 'side': side, 'ref': r['ref'],
                    'from_id': r['from_id'], 'to_id': r['to_id'], 'direction': r['direction'],
                    'labels': r['labels'], 'source': r['source'], 'target': r['target'], 'valid_typed_endpoints': good})
            for i in p['issues']:
                problem_rows.append({'task': key, 'project': project, 'side': side, 'annotator': p['annotator'], **i})
        text = raw['data']['text']
        privacy = [{'field': m.group(1), 'start': m.start(), 'end': m.end()} for m in re.finditer(
            r'(姓名|身份证(?:号)?|联系电话|住院号|床号)[ \t]*[:：][ \t]*([^\n\r，。;；]+)', text)
            if not re.fullmatch(r'\[[^\]]*\]|[_*Xx某/\s]+', m.group(2).strip())]
        windows = display_windows(text, [(e['start'], e['end']) for p in parsed.values() for e in p['entities']])
        task = {'key': key, 'project': project, 'ls_task_id': raw['id'], 'business_task_id': str(raw['data'].get('task_id', '')),
            'text_length': len(text), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'metadata_identifier_fields': [f for f in ('patient_id', 'serial_number', 'patient_group_id') if raw['data'].get(f)],
            'privacy_candidates': privacy, 'annotations': parsed, 'review_rows': rows, 'display_windows': windows,
            'entity_metric': metric(ka, kb), 'case_A': ca, 'case_B': cb, 'pair_group': pair_key,
            'fully_equal': all(r['status'] in CONSISTENT for r in rows if r['kind'] != '病例复核建议')}
        output_tasks.append(task)
        task_rows.append({k: task[k] for k in ('key', 'project', 'ls_task_id', 'business_task_id', 'pair_group',
            'text_length', 'text_sha256', 'case_A', 'case_B', 'entity_metric', 'fully_equal')})
    labels = label_table(overall, next(iter(schemas.values()))['labels'], cards)
    fields = {f: attribute_statistics(overall['fields'][f], overall['attribute_pairs'][f]) for f in FIELD_NAMES}
    for f in fields:
        fields[f]['both_nonempty_kappa'] = None  # Do not pool nominal kappa across different rater pairs.
        fields[f]['coverage'] = dict(overall['field_coverage'][f])
    group_metrics = {g: accumulator_metrics(acc, {u: s for u, s in zip(g.split(':')[1].split('/'), ('A', 'B'))})
                     for g, acc in groups.items()}
    project_metrics = {}
    for project, acc in projects.items():
        own_groups = [g for g in groups if g.split(':')[0] == project]
        project_metrics[project] = accumulator_metrics(acc, group_metrics[own_groups[0]]['annotators'] if len(own_groups) == 1 else {}, len(own_groups) == 1)
        project_metrics[project]['pair_groups'] = own_groups
    text_groups = defaultdict(list)
    for t in output_tasks: text_groups[t['text_sha256']].append(t['key'])
    case = kappa(overall['case_pairs'])
    if case: case = {**case, 'kappa': None}  # Cross-group pooled kappa is deliberately not interpreted.
    summary = {'tasks': len(tasks), 'annotations': len(tasks)*2, 'result_rows': len(inventory), 'review_rows': len(all_rows),
        'pending_rows': sum(r['status'] not in CONSISTENT for r in all_rows), 'fully_equal_tasks': sum(t['fully_equal'] for t in output_tasks),
        'case': case, 'entities': metric(*overall['entities']), 'projects': project_metrics, 'issues': dict(overall['issues']),
        'duplicate_text_groups': [g for g in text_groups.values() if len(g)>1], 'privacy_candidates': sum(bool(t['privacy_candidates']) for t in output_tasks),
        'result_types': dict(overall['result_types']), 'relations': accumulator_metrics(overall)['typed_relation'],
        'relation_endpoints': metric(*overall['endpoints']), 'relations_not_fully_typed': overall['invalid_typed'],
        'label_coverage': sum(l['A']+l['B']>0 for l in labels), 'updated_by_other_annotations': sum(a['updated_by_other'] for a in annotation_rows)}
    relations = relation_table(overall, next(iter(schemas.values()))['relations'])
    detailed = {'pair_groups': group_metrics, 'project_labels': {p: label_table(a, next(iter(schemas.values()))['labels'], {}) for p, a in projects.items()},
                'group_fields': {g: {f: attribute_statistics(a['fields'][f], a['attribute_pairs'][f]) | {'coverage': dict(a['field_coverage'][f])} for f in FIELD_NAMES} for g, a in groups.items()},
                'group_relations': {g: relation_table(a, next(iter(schemas.values()))['relations']) for g,a in groups.items()},
                'project_relations': {p: relation_table(a, next(iter(schemas.values()))['relations']) for p,a in projects.items()},
                'relations': relations}
    tables = {'tasks': task_rows, 'annotations': annotation_rows, 'entities': entity_rows, 'attributes': attribute_rows,
              'relations': relation_rows, 'issues': problem_rows,
              'choice_value_distribution':[{'project':p,'pair_group':g,'side':s,'field':f,'normalized_values':json.loads(v),'result_count':n}
                  for (p,g,s,f,v),n in sorted(choice_counts.items())]}
    return {'summary': summary, 'tasks': output_tasks, 'inventory': inventory, 'review_rows': all_rows,
            'labels': labels, 'fields': fields, 'detailed': detailed}, tables


def label_table(acc, labels, cards):
    rows = []
    for label in labels:
        c = acc['labels'][label]; n = c['A']+c['B']
        rows.append({'label': label, 'A': c['A'], 'B': c['B'], 'exact': c['exact'],
            'A_only': c['A']-c['exact'], 'B_only': c['B']-c['exact'], 'f1': 2*c['exact']/n if n else None,
            'coverage': '出现' if n else '本批未出现，不能评价', 'card': cards.get(label, {})})
    return rows


def relation_table(acc, relation_types):
    return [{'type': r, **metric(*acc['typed_by_type'][r]),
             'evaluation': '类型记录不完整，不能评价全量类型一致性' if acc['invalid_typed'] else '已记录类型子集严格F1',
             'full_f1': None if acc['invalid_typed'] else metric(*acc['typed_by_type'][r])['symmetric_f1']}
            for r in relation_types]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')


def write_csv(path, rows, columns=None):
    if columns is None: columns = list(rows[0]) if rows else ['status']
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); writer.writeheader()
        for row in rows:
            values = {k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            # Prevent spreadsheet formula execution from source text.
            writer.writerow({k: ("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for k,v in values.items()})


def flat_pair_metrics(group, m):
    c = m['case'] or {}
    return {'group':group,'tasks':m['tasks'],'result_rows':m['result_rows'],
        'case_valid_pairs':c.get('n',0),'case_agree':c.get('agree',0),'case_agreement':c.get('agreement'),
        'case_kappa':c.get('kappa'), **{'entity_'+k:v for k,v in m['entity'].items()},
        **{'relation_endpoints_'+k:v for k,v in m['relation_endpoints_only'].items()},
        'relations_not_fully_typed':m['relations_not_fully_typed'],
        'full_typed_relation_f1':(m['typed_relation'] or {}).get('symmetric_f1')}


def flat_attribute_metrics(field, s):
    c = s.get('both_nonempty_kappa') or {}
    return {'field':field,'name':FIELD_NAMES[field],**{k:v for k,v in s.items() if k not in ('both_nonempty_kappa','coverage')},
        'both_nonempty_pairs':c.get('n') if c else None,'both_nonempty_agree':c.get('agree') if c else None,
        'both_nonempty_kappa':c.get('kappa'),**{k:s.get('coverage',{}).get(k,0) for k in ('A_applicable','A_filled','B_applicable','B_filled')}}


def run(input_dir, output_root, run_id, candidate_dir, fallback_schema, guide, dictionary):
    input_dir, output_root = Path(input_dir).resolve(), Path(output_root).resolve()
    paths = sorted(p for p in input_dir.glob('*.json') if p.is_file())
    if not paths: raise ValueError('No JSON exports found')
    dirs = {s: output_root/s/run_id for s in ('data_preparation', 'annotation_analysis', 'adjudication', 'delivery')}
    if any(p.exists() for p in dirs.values()): raise ValueError('Run directory exists; use a new run_id')
    tasks, sources, lineage, blocked = integrate(paths)
    if not tasks: raise ValueError('No unambiguous double-annotated task')
    for p in dirs.values(): p.mkdir(parents=True)
    prep, analysis, review, delivery = (dirs[s] for s in ('data_preparation','annotation_analysis','adjudication','delivery'))
    write_json(prep/'merged_tasks.json', tasks)
    write_json(prep/'source_lineage.json', lineage)
    write_csv(prep/'source_lineage.csv', lineage)
    write_csv(prep/'blocked_tasks.csv', blocked)
    schemas, schema_sources, schema_paths = {}, [], {}
    (prep/'schema_candidates').mkdir()
    for project in sorted({str(t['project']) for t in tasks}):
        candidate = Path(candidate_dir)/f'project-{project}.candidate.xml' if candidate_dir else Path(fallback_schema)
        if not candidate.is_file(): candidate = Path(fallback_schema)
        snapshot = prep/'schema_candidates'/f'project_{project}.candidate.xml'
        shutil.copy2(candidate, snapshot); schemas[project] = load_schema(snapshot); schema_paths[project] = snapshot
        schema_sources.append({'project': project, 'source': str(candidate.resolve()), 'snapshot': str(snapshot),
                               'sha256': digest(snapshot), 'verified': False, 'status': 'candidate_actual_project_configuration_pending'})
    schema = next(iter(schemas.values()))
    if any(canonical(s) != canonical(schema) for s in schemas.values()):
        raise ValueError('Schema contracts differ; group analyses by compatible contract')
    cards = dictionary_cards(dictionary, schema['labels'])
    payload, tables = analyse(tasks, schemas, cards)
    for source in sources:
        source['schema_file'] = str(schema_paths[source['project']]); source['schema_sha256'] = digest(schema_paths[source['project']])
    limitations = ['全部项目配置仅作候选合同，尚需负责人核实实际标注时XML及版本。',
        '228的1/92与92/95按实际人员分组，1号账号角色待核实，不自动映射人员。',
        '双人一致可能同错；原始F1是跨度和标签的对称一致性，不是医学准确率。',
        '属性只在单一精确匹配实体上比较，双方空白单列，未匹配实体属性仍需复核。',
        '不同人员组合不合并解释κ，单例κ不作解释。关系类型缺失不推断，端点F1不代替完整类型F1。',
        '展示遮盖不等于完整去标识化，病历片段只用于受控访问。',
        '尚未进行医学签署、Label Studio回标或训练金标冻结。']
    context = {'guide_sha256': digest(guide), 'dictionary_sha256': digest(dictionary), 'notes_sha256': None,
               'integration_sha256': digest(prep/'source_lineage.json'), 'schema_status': 'candidate_unverified'}
    payload.update(version='1.0', sources=sources, review_context=context, schema=schema,
        guide={'file': str(guide), 'sha256': digest(guide)}, dictionary={'file': str(dictionary), 'sha256': digest(dictionary)},
        limitations=limitations, batch_context={'configuration_status': '候选配置待核实',
            'notice': '本批配置尚未确认。228含两种实际人员组合，1号账号角色待核实。当前可开展逐项复核，最终参考冻结前须完成输入确认。',
            'statistics_url': f'../../annotation_analysis/{run_id}/statistics.html'},
        field_names=FIELD_NAMES)
    write_json(review/'audit.json', payload)
    (review/'review.html').write_text(build_html(payload), 'utf-8')
    guide_doc = ROOT/'documentation/double_annotation_adjudication/double-annotation-adjudication.md'
    try:
        guide_link = Path(os.path.relpath(guide_doc, review)).as_posix()
    except ValueError:  # Windows output may live on a different drive.
        guide_link = guide_doc.as_uri()
    (review/'核查报告.md').write_text(build_report(payload, guide_link), 'utf-8')
    decisions = {'version': '1.0', 'sources': sources, 'review_context': context,
        'decisions': [{k: r[k] for k in ('id','task','kind')} | {'status':'待裁决','final_answer':'','evidence_offsets':[],
                      'rule_ref':'','reason_category':'','reason':'','adjudicator':'','reviewed_at':'','needs_back_annotation':False} for r in payload['review_rows']]}
    write_json(review/'裁决模板.json', decisions)
    write_json(review/'标签核查签署模板.json', {'sources':sources,'review_context':context,
        'labels':[{'label':l['label'],'A_count':l['A'],'B_count':l['B'],'coverage':l['coverage'],'status':'待核查',
                   'reviewed_tasks':[],'new_missing_mentions':[],'review_result':'','rule_ref':l['card'].get('section',''),
                   'adjudicator':'','reviewed_at':''} for l in payload['labels']]})
    for name, rows in tables.items(): write_csv(analysis/(name+'.csv'), rows)
    write_csv(analysis/'label_agreement.csv', payload['labels'], ['label','A','B','exact','A_only','B_only','f1','coverage'])
    write_csv(analysis/'attribute_agreement.csv', [flat_attribute_metrics(f,s) for f,s in payload['fields'].items()])
    write_csv(analysis/'pair_agreement.csv', [flat_pair_metrics(g,m) for g,m in payload['detailed']['pair_groups'].items()])
    write_csv(analysis/'project_agreement.csv', [flat_pair_metrics(p,m) for p,m in payload['summary']['projects'].items()])
    write_csv(analysis/'project_label_agreement.csv', [{'project':p,**l} for p, ls in payload['detailed']['project_labels'].items() for l in ls],
              ['project','label','A','B','exact','A_only','B_only','f1','coverage'])
    write_csv(analysis/'pair_attribute_agreement.csv', [{'group':g,**flat_attribute_metrics(f,s)} for g,fs in payload['detailed']['group_fields'].items() for f,s in fs.items()])
    write_csv(analysis/'relation_agreement.csv', payload['detailed']['relations'])
    write_csv(analysis/'pair_relation_agreement.csv', [{'group':g,**r} for g,rs in payload['detailed']['group_relations'].items() for r in rs])
    write_csv(analysis/'project_relation_agreement.csv', [{'project':p,**r} for p,rs in payload['detailed']['project_relations'].items() for r in rs])
    write_csv(review/'case_disagreements.csv', [{'task':t['key'],'project':t['project'],'pair_group':t['pair_group'],
        'A':t['case_A'],'B':t['case_B'],'status':'待裁决'} for t in payload['tasks'] if t['case_A']!=t['case_B']])
    write_csv(review/'review_queue.csv', [{k:r[k] for k in ('id','task','project','pair_group','kind','status','rule')} for r in payload['review_rows']])
    stats = {k: payload[k] for k in ('summary','labels','fields','detailed','limitations','field_names')}
    stats['integration'] = {'source_files':len(paths),'input_task_rows':sum(s['task_rows'] for s in sources),
        'unique_task_keys':len(lineage),'included_tasks':len(tasks),'removed_duplicate_task_rows':sum(s['task_rows'] for s in sources)-len(lineage),
        'blocked_tasks':len(blocked),'source_lineage':lineage}
    stats['tasks'] = tables['tasks']; stats['review_url'] = f'../../adjudication/{run_id}/review.html'
    stats['schemas'] = schema_sources
    write_json(analysis/'statistics.json', stats)
    template = (Path(__file__).parent/'templates/batch_statistics.html').read_text('utf-8')
    (analysis/'statistics.html').write_text(template.replace('__PAYLOAD__', json.dumps(stats,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')), 'utf-8')
    # Dynamic landing page links stage outputs, never embeds raw identifiers.
    (delivery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>试标整合与裁决入口</title><style>body{font:18px system-ui;max-width:850px;margin:60px auto;line-height:2}a{color:#087d88}</style>'
        f'<h1>试标整合、统计与双人裁决</h1><p>{len(paths)}份输入，{len(tasks)}例病例，{payload["summary"]["annotations"]}份标注。候选配置待确认，尚无医学签署。</p>'
        f'<p><a href="../../annotation_analysis/{run_id}/statistics.html">查看完整统计图表与明细</a></p>'
        f'<p><a href="../../adjudication/{run_id}/review.html">进入V2双人裁决工作台</a></p>'
        f'<p><a href="../../adjudication/{run_id}/裁决模板.json">裁决JSON模板</a>　<a href="../../adjudication/{run_id}/标签核查签署模板.json">49标签全文查漏签署模板</a></p>'
        '<p>每个浏览器独立保存草稿。请定期导出JSON；当前工具不自动回标或生成训练金标。</p>', 'utf-8')
    report = ['# 试标整合及一致性统计', '', f'本批输入{len(paths)}份，含{sum(s["task_rows"] for s in sources)}条文件任务记录。去重后{len(tasks)}例、{len(tasks)*2}份标注；重复任务记录只计一次。',
        '', '## 指标口径', '', '病例一致率=相同有效结论/有效病例对。κ仅按固定人员组合解释。严格实体F1=2×精确共同/(A+B)，任务、跨度、标签和重复次数均参与匹配。',
        '属性非空一致率=非空相同/至少一方填写槽位，双方空白单列。属性κ只在双方有有效单选的子集计算，分母另列，单例不解释。',
        '关系端点F1包含方向和端点实体标签；缺类型不推断，不以端点F1代替完整关系F1。全部原始错误和同错仍保留。',
        '', '## 人员组合', '', '| 项目/人员 | 例数 | 病例一致 | κ | 实体F1 | 端点F1 |', '|---|---:|---:|---:|---:|---:|']
    for g,m in payload['detailed']['pair_groups'].items():
        c=m['case']; agreement=f'{c["agree"]}/{c["n"]}' if c else '缺有效结论'; k=c['kappa'] if c and c['n']>1 else '不可解释'
        report.append(f'| {g} | {m["tasks"]} | {agreement} | {k} | {m["entity"]["symmetric_f1"]} | {m["relation_endpoints_only"]["symmetric_f1"]} |')
    report.extend(['','## 使用顺序','','1. 数据负责人确认各项目XML和1号账号角色，核对其他账号编辑记录。',
        '2. 质量人员处理偏移、引用、关系类型和重复问题；候选问题逐条回读，不批量删改。',
        '3. 医学裁决员先查病例分歧，再查一致项；逐个49标签全文查漏，然后核查9属性和5关系。',
        '4. 在V2页面填写答案、证据、规则、理由、人员和时间，导出裁决JSON。标签全文查漏在独立签署模板中登记。',
        '5. 负责人按签署结论修正标注副本、回导出复检。原始双标指标不更新为裁决后指标。', '', '## 限制', '']+limitations)
    (analysis/'统计说明.md').write_text('\n'.join(report)+'\n','utf-8')
    (review/'使用说明.md').write_text('# 本批裁决工作台使用说明\n\n'
        '从统计页面定位项目、标签或病例后进入review.html，按病例、49实体标签、9属性和5关系逐项核查。双方一致项也须阅读全文复核。\n\n'
        '填写最终答案、证据偏移、规则、原因、裁决人、时间和回标要求，确认后导出裁决JSON。采用A/B只是候选，不自动签署。\n\n'
        '浏览器独立存草稿，不自动同步。每次结束先导出，再把文件保存在受控目录。移换地址时先导出再导入，不强行导入旧批记录。\n\n'
        '49标签全文查漏请填写独立标签核查签署模板，逐标签记录已读病例、双方漏标和未覆盖状态。\n\n'
        '配置和1号账号角色待核实。当前为待人工复核材料，没有自动医学裁决、Label Studio回写或训练金标转换。\n','utf-8')
    source_unchanged = all(digest(s['path']) == s['sha256'] for s in sources)
    assert source_unchanged
    assert len({t['key'] for t in payload['tasks']}) == len(tasks)
    assert len({r['id'] for r in payload['review_rows']}) == len(payload['review_rows'])
    assert len(inventory := payload['inventory']) == sum(len(a.get('result',[])) for t in tasks for a in t['annotations'] if not a.get('was_cancelled'))
    code_paths = [Path(__file__), ROOT/'emr_annotation/annotation_analysis/double_annotations.py', ROOT/'emr_annotation/annotation/schema.py',
                  ROOT/'emr_annotation/adjudication/templates/double_annotation_review.html', Path(__file__).parent/'templates/batch_statistics.html']
    manifest = {'batch_id':input_dir.name,'stage':'data_preparation+annotation_analysis+adjudication+delivery','run_id':run_id,
        'created_at':datetime.now().astimezone().isoformat(),'input_directory':str(input_dir),'sources':sources,'schemas':schema_sources,
        'guide_sha256':digest(guide),'dictionary_sha256':digest(dictionary),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        'code_files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in code_paths],
        'parameters':{'input_dir':str(input_dir),'run_id':run_id,'candidate_dir':str(candidate_dir),'fallback_schema':str(fallback_schema),
                      'guide':str(guide),'dictionary':str(dictionary),'duplicate_policy':'equal data and submitted annotation content only; no latest-answer selection'},
        'integration':stats['integration'] | {'source_lineage':str(prep/'source_lineage.json')},
        'validation':{'source_hashes_unchanged':source_unchanged,'unique_task_and_review_ids':True,'raw_result_rows_reconciled':True,
                      'actual_project_schemas_verified':False,'medical_decisions_signed':0},
        'outputs':[{'path':str(p.relative_to(output_root)),'sha256':digest(p),'bytes':p.stat().st_size} for folder in dirs.values() for p in folder.rglob('*') if p.is_file()]}
    write_json(delivery/'manifest.json', manifest)
    return {'output':str(delivery/'index.html'),'summary':payload['summary'],'integration':stats['integration'] | {'source_lineage':'see source_lineage.json'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--fallback-schema', type=Path, required=True)
    parser.add_argument('--guide',type=Path,default=ROOT/'documentation/annotation/annotation_guide_v2.1.1.md')
    parser.add_argument('--dictionary',type=Path,default=ROOT/'documentation/annotation/label_dictionary_v2.1.1.md')
    args=parser.parse_args(argv)
    result=run(args.input_dir,args.output_root,args.run_id,args.candidate_dir,args.fallback_schema,args.guide,args.dictionary)
    print(json.dumps({'output':result['output'],'integration':result['integration'],
        'summary':{k:v for k,v in result['summary'].items() if k not in ('projects','case','issues')},
        'case_agreement':result['summary']['case']},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
