"""Audit known per-region detection attributes without declaring unknown slots negative."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

from emr_annotation.training_data.schema_multitask import sha256,write_json

FIELDS=('test_result','specimen_type','detection_method')


def read(p):return json.loads(Path(p).read_text('utf-8'))


def audit(reference,predictions,schema):
    pred={r['task_id']:r for r in predictions};results={};joint=[]
    for field in FIELDS:
        counts=Counter();correct=Counter();matched=0;confusion=Counter()
        for row in reference:
            entities={(e['start'],e['end'],e['label']):e for e in pred[row['task_id']]['entities']}
            for ent in row['entities']:
                target=ent.get('attributes',{}).get(field)
                if target is None or ent['label'] not in schema['fields'][field]['scope']:continue
                counts[target]+=1;p=entities.get((ent['start'],ent['end'],ent['label']))
                answer=p.get('attributes',{}).get(field,'missing_attribute') if p else 'missing_entity'
                matched+=bool(p);correct[target]+=answer==target;confusion[(target,answer)]+=1
        total=sum(counts.values());hits=sum(correct.values())
        results[field]=dict(known_slots=total,matched_entities=matched,correct=hits,accuracy=hits/total if total else None,
                           majority_option=max(counts,key=counts.get) if counts else None,
                           majority_option_reference_fraction=max(counts.values())/total if total else None,
                           choices=[dict(option=o,support=counts[o],correct=correct[o],known_slot_recall=correct[o]/counts[o] if counts[o] else None) for o in schema['fields'][field]['options']],
                           confusion=[dict(reference=a,prediction=b,count=n) for (a,b),n in confusion.items()])
    for row in reference:
        entities={(e['start'],e['end'],e['label']):e for e in pred[row['task_id']]['entities']}
        for e in row['entities']:
            if not all(f in e.get('attributes',{}) and e['label'] in schema['fields'][f]['scope'] for f in FIELDS):continue
            p=entities.get((e['start'],e['end'],e['label']))
            joint.append(bool(p and all(p.get('attributes',{}).get(f)==e['attributes'][f] for f in FIELDS)))
    return dict(fields=results,joint_known_record=dict(support=len(joint),correct=sum(joint),accuracy=sum(joint)/len(joint) if joint else None),
                metric_scope='Known reference attribute slots/records only. Missing entity counts as wrong. Not full predicted-record precision/F1; unannotated reference slots remain unknown.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','predictions','schema','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new audit output')
    result=audit(read(a.reference),read(a.predictions),read(a.schema))
    result.update(status='completed',created_at=datetime.now(timezone.utc).isoformat(),inputs={str(f.resolve()):sha256(f) for f in (a.reference,a.predictions,a.schema)},code_sha256=sha256(Path(__file__)))
    write_json(a.output,result);print(json.dumps(dict(joint=result['joint_known_record'],fields={f:r['accuracy'] for f,r in result['fields'].items()}),ensure_ascii=False))


if __name__=='__main__':main()
