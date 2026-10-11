"""Prepare explicitly provisional annotation views without clinical adjudication."""
from collections import Counter
from pathlib import Path
import hashlib
import json

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.annotation_analysis.double_annotations import parse_annotation
from emr_annotation.annotation_analysis.trial_batch import integrate
from emr_annotation.training_data.split_reference import split_reference


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def relation_type_allowed(source_label, target_label, relation_type, schema):
    """Schema/guide topology only; does not adjudicate clinical correctness."""
    indicators = set(schema['controls'].get('measure_entities', []))
    components = set(schema['controls'].get('measure_labels', []))
    if relation_type not in schema['relations']:
        return False
    if relation_type == '测量':
        return source_label in indicators and target_label in components
    return target_label == '时间表达' and source_label not in components | {'时间表达'}


def project_typed_relation(relation, region_entities, schema):
    """Retain only a unique compatible projection of an explicitly typed edge."""
    labels = relation.get('labels', [])
    if (len(labels) != 1 or labels[0] not in schema['relations'] or
            relation.get('direction') not in ('left', 'right') or
            relation.get('from_id') == relation.get('to_id')):
        return None
    sources = region_entities.get(relation['from_id'], [])
    targets = region_entities.get(relation['to_id'], [])
    if relation['direction'] == 'left':
        sources, targets = targets, sources
    candidates = {(s['id'], t['id']): (s, t) for s in sources for t in targets
                  if relation_type_allowed(s['label'], t['label'], labels[0], schema)}
    if len(candidates) != 1:
        return None
    source, target = next(iter(candidates.values()))
    return {'from_id': source['id'], 'to_id': target['id'], 'type': labels[0], 'direction': 'right',
            'source_relation_ref': relation['ref'],
            'projection_policy': 'unique_schema_compatible_endpoint_pair_from_original_typed_region_edge'}


def prepare(input_dir, schema_path, output_dir, seed=42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    schema = load_double_annotation_schema(schema_path)
    tasks, sources, lineage, blocked = integrate(Path(input_dir).glob('project-*.json'))
    records, issues, excluded, support = [], [], Counter(), Counter()
    for task in tasks:
        patient = task['data'].get('patient_group_id') or task['data'].get('patient_id')
        if patient is None or not str(patient).strip():
            raise ValueError('Stable patient key required')
        for ann in task['annotations']:
            if ann.get('was_cancelled'):
                continue
            key = f"p{task['project']}_t{task['id']}_a{ann['id']}"
            parsed = parse_annotation(task, ann, schema, 'raw', key)
            issues.extend({'task_id': key, **i} for i in parsed['issues'])
            bad_entities = {i['ref'] for i in parsed['issues'] if i['code'] in
                            ('invalid_offsets', 'span_text_mismatch', 'label_control_mismatch', 'region_id_span_collision')}
            entities, id_to_entities = [], {}
            seen = set()
            for ent in parsed['entities']:
                if ent['ref'] in bad_entities:
                    excluded['invalid_entities'] += 1
                    continue
                span = (ent['start'], ent['end'], ent['label'])
                if span in seen:
                    excluded['duplicate_entities'] += 1
                    continue
                seen.add(span)
                attrs = {}
                for field, vals in ent['attributes'].items():
                    rule = schema['fields'].get(field, {})
                    attr_issues = {i['ref'] for i in parsed['issues'] if i['code'] in
                                   ('attribute_span_mismatch', 'duplicate_attribute', 'invalid_choice', 'attribute_not_applicable')}
                    if (ent['label'] in rule.get('scope', []) and len(vals) == 1 and vals[0] in rule['values']
                            and not (set(ent['attribute_refs'].get(field, [])) & attr_issues)):
                        attrs[field] = vals[0]
                        support['attribute:' + field] += 1
                    else:
                        excluded['invalid_attribute_slots'] += 1
                entity = {'id': f'e{len(entities)}', 'start': ent['start'], 'end': ent['end'],
                          'label': ent['label'], 'attributes': attrs}
                entities.append(entity)
                id_to_entities.setdefault(ent['id'], []).append(entity)
                support['entity:' + ent['label']] += 1
            relations = []
            for rel in parsed['relations']:
                projected = project_typed_relation(rel, id_to_entities, schema)
                if projected is None:
                    excluded['invalid_relations'] += 1
                    continue
                relations.append(projected)
                support['relation:' + rel['labels'][0]] += 1
            case_values = [v for c in parsed['case'] for v in c['values']]
            case = case_values[0] if len(case_values) == 1 and case_values[0] in schema['fields']['case_decision']['values'] else None
            if case:
                support['case:' + case] += 1
            records.append({'task_id': key, 'patient_id': str(patient), 'text': task['data']['text'],
                            'project': task['project'], 'source_task_id': task['id'], 'annotation_id': ann['id'],
                            'annotator': ann['completed_by'], 'entities': entities, 'relations': relations,
                            'case_decision': case, 'case_choices': {'case_decision': case} if case else {},
                            'ner_supervised': not bad_entities,
                            'relations_supervised': bool(relations), 'relations_supervision': 'positive_only',
                            'reference_status': 'unadjudicated_annotation_view'})
    groups = schema['controls']
    for record in records:
        if record['case_decision'] is None:
            del record['case_decision']
    splits, manifest = split_reference(records, groups, (0.7, 0.15, 0.15), seed)
    write_json(out / 'reference.json', records)
    for name, rows in splits.items():
        write_json(out / (name + '.json'), rows)
    write_json(out / 'split_manifest.json', manifest)
    write_json(out / 'issues.json', issues)
    write_json(out / 'source_lineage.json', lineage)
    Path(out / 'label_config.xml').write_bytes(Path(schema_path).read_bytes())
    compact_schema = {'labels': schema['labels'], 'controls': groups, 'relations': schema['relations'],
                      'fields': {k: {**v, 'options': list(v['values'])} for k, v in schema['fields'].items()},
                      'schema_sha256': sha256(schema_path)}
    write_json(out / 'schema.json', compact_schema)
    audit = {'status': 'provisional_not_gold', 'sources': sources, 'unique_tasks': len(tasks),
             'annotation_views': len(records), 'blocked': blocked, 'exclusions': dict(excluded),
             'issue_counts': dict(Counter(i['code'] for i in issues)), 'support': dict(support),
             'split_counts': {k: len(v) for k, v in splits.items()},
             'patient_counts': {k: len({r['patient_id'] for r in v}) for k, v in splits.items()},
             'policy': 'Both submitted annotator views retained; patient and identical text components never split. '
                       'Invalid entity views have NER loss masked; valid attributes retained. Missing/invalid '
                       'relation annotations masked, never converted to negative relations. Explicit typed region edges '
                       'may yield a unique Schema-compatible entity pair; each projection preserves its original relation reference. '
                       'Relation supervision is positive-only; unobserved pairs are unknown. No medical adjudication.',
             'source_hashes_unchanged': all(sha256(s['path']) == s['sha256'] for s in sources)}
    write_json(out / 'preparation_audit.json', audit)
    return audit
