"""Freeze AI sample references using existing case/patient split boundaries."""
from collections import Counter
from pathlib import Path
import argparse
import json

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.training_data.schema_multitask import sha256, write_json
from emr_annotation.adjudication.ai_sample import validate_samples


def prepare(reference_path, schema_path, output_dir, preserved_split_dir):
    source = Path(reference_path)
    schema = load_double_annotation_schema(schema_path)
    records = json.loads(source.read_text('utf-8'))
    validate_samples(records, schema)
    if any(r.get('reference_status') != 'ai_sample_not_expert_gold' or not r.get('case_reviewed_by_ai') for r in records):
        raise ValueError('Only explicitly reviewed AI sample references accepted')
    if len({r['task_id'] for r in records}) != len(records):
        raise ValueError('Duplicate task')
    assignment, prior, hashes = {}, {}, {}
    names = ('train','validation','test')
    for name in names:
        path = Path(preserved_split_dir) / (name+'.json')
        hashes[str(path.resolve())] = sha256(path)
        for r in json.loads(path.read_text('utf-8')):
            key = (r['project'],r['source_task_id'])
            if key in assignment and assignment[key] != name:
                raise ValueError('Previous case views crossed splits')
            assignment[key] = name
            prior[key] = r
    if set(assignment) != {(r['project'],r['source_task_id']) for r in records}:
        raise ValueError('Sample scope must exactly match preserved split cases')
    splits = {name:[] for name in names}
    patient_owner, text_owner = {}, {}
    for row in records:
        key = (row['project'],row['source_task_id'])
        name = assignment[key]
        if row['text'] != prior[key]['text'] or row['patient_id'] != prior[key]['patient_id']:
            raise ValueError('Changed text or patient grouping')
        for value, owners in ((row['patient_id'],patient_owner),(row['text'],text_owner)):
            if owners.setdefault(value,name) != name:
                raise ValueError('Patient or exact text split leakage')
        row['case_choices'] = {'case_decision':row['case_decision']}
        splits[name].append(row)
    summaries = {name:{'tasks':len(rows),'patients':len({r['patient_id'] for r in rows}),
                       'unique_texts':len({r['text'] for r in rows}),
                       'case_decision_counts':dict(Counter(r['case_decision'] for r in rows)),
                       'reference_entities':sum(len(r['entities']) for r in rows),
                       'ner_supervised_cases':sum(r['ner_supervised'] for r in rows)} for name,rows in splits.items()}
    coverage = {label:{'entities':{name:sum(e['label']==label for r in rows for e in r['entities']) for name,rows in splits.items()}} for label in schema['labels']}
    manifest = {'tasks':len(records),'patients':len(patient_owner),'unique_texts':len(text_owner),
                'grouping':'preserved_patient_and_identical_text_components', 'splits':summaries,'label_coverage':coverage,
                'verified_no_patient_overlap':True,'verified_no_exact_text_overlap':True,
                'near_duplicate_check':'not_performed','preserved_split_files':hashes,
                'source_sha256':sha256(source),'label_config_sha256':sha256(schema_path),
                'assignments':[{'task_id':r['task_id'],'split':name,'text_sha256':r['text_sha256']} for name,rows in splits.items() for r in rows]}
    support = Counter('entity:'+e['label'] for r in records for e in r['entities'])
    support.update('attribute:'+f for r in records for e in r['entities'] for f in e['attributes'])
    support.update('relation:'+rel['type'] for r in records for rel in r['relations'])
    support.update('case:'+r['case_decision'] for r in records)
    issues = [{'task_id':r['task_id'],**i} for r in records for i in r['issues']]
    audit = {'status':'ai_sample_not_expert_gold','unique_tasks':len(records),'annotation_views':len(records),
             'sources':[{'path':str(source.resolve()),'sha256':sha256(source),'task_rows':len(records)}],
             'source_hashes_unchanged':True,'blocked':[], 'exclusions':{'ner_masked_cases':sum(not r['ner_supervised'] for r in records)},
             'issue_counts':dict(Counter(i['code'] for i in issues)), 'support':dict(support),
             'split_counts':{k:len(v) for k,v in splits.items()},'patient_counts':{k:v['patients'] for k,v in summaries.items()},
             'expert_signed':0,'policy':'One AI sample per case, with case-level AI reasoning and rule-assisted point reconciliation. '
                      'Incomplete/uncertain source points mask case NER loss and entity evaluation. Known attribute slots retained. '
                      'Unobserved relation pairs remain unknown. Existing patient/text assignments preserved. '
                      'Sample NER assumes proposed mentions complete only in included cases; this is weak supervision, not verified exhaustiveness. '
                      'This development batch has already been inspected; the test is not an independent medical blind test.'}
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in splits.items():
        write_json(out/(name+'.json'),rows)
    write_json(out/'reference.json',records)
    write_json(out/'split_manifest.json',manifest)
    write_json(out/'preparation_audit.json',audit)
    write_json(out/'issues.json',issues)
    (out/'label_config.xml').write_bytes(Path(schema_path).read_bytes())
    write_json(out/'schema.json',{'labels':schema['labels'],'controls':schema['controls'],'relations':schema['relations'],
                                 'fields':{k:{**v,'options':list(v['values'])} for k,v in schema['fields'].items()},'schema_sha256':sha256(schema_path)})
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--schema',type=Path,required=True)
    p.add_argument('--preserve-splits-from',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    print(prepare(a.reference,a.schema,a.output_dir,a.preserve_splits_from)['split_counts'])


if __name__=='__main__':
    main()
