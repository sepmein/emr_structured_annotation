"""Verify frozen human decisions, split isolation and saved hybrid predictions.

Stdlib only; this does not certify clinical labels or model calibration.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from emr_annotation.training_data.schema_multitask import sha256,write_json
from emr_annotation.evaluation.clinical_evidence import extract


def read(path):return json.loads(Path(path).read_text('utf-8'))


def verify(freeze_dir,data_dir,extractor_dir,fusion_dir,predictions,review_dir,output,*,
           case_key='fusion',extractor_metrics=None,saved_test_predictions=None,comparison_path=None):
    if output.exists():raise ValueError('Verification output already exists')
    freeze=read(freeze_dir/'freeze_manifest.json')
    for name,expected in freeze['outputs'].items():
        assert sha256(freeze_dir/name)==expected,('Changed frozen output',name)
    reference=read(freeze_dir/'reference.json');byid={r['task_id']:r for r in reference}
    assert len(byid)==freeze['cases']
    with sqlite3.connect((review_dir/'review_state.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as con:
        saved=con.execute('SELECT task_id,revision,document FROM drafts').fetchall()
    statuses=Counter();points=Counter()
    for tid,revision,encoded in saved:
        doc=json.loads(encoded);statuses[doc['case_status']]+=1;points.update(x['status'] for x in doc['point_reviews'].values())
        assert tid in byid and doc['case_status']=='approved'
        r=byid[tid];assert revision==r['review_revision']
        for field in ('entities','relations','case_decision'):assert doc['candidate'][field]==r[field],(tid,field)
        assert doc['point_reviews']==r['human_point_reviews']
    assert len(saved)==len(reference) and not points['pending'] and not points['deferred']
    schema=read(data_dir/'schema.json')
    splits={n:read(data_dir/(n+'.json')) for n in ('train','validation','test')}
    owners={};texts={};synthetic=0;counts={};real_assignment={}
    for name,rows in splits.items():
        counts[name]=dict(real=0,synthetic=0,total=len(rows))
        for row in rows:
            is_synthetic=row.get('synthetic',False);counts[name]['synthetic' if is_synthetic else 'real']+=1
            if is_synthetic:
                assert name=='train';synthetic+=1
            else:
                assert owners.setdefault(row['patient_id'],name)==name
                assert texts.setdefault(row['text'],name)==name
                r=byid[row['task_id']];real_assignment[row['task_id']]=name
                for field in ('text','entities','relations','case_decision'):assert row[field]==r[field]
            target=extract(row['text'],row['entities'],row['relations'],schema)
            for field in ('syndromes','pathogens','pathogen_mask'):assert row['evidence_targets'][field]==target[field]
    assert synthetic==1000 and set(real_assignment)==set(byid)
    inferred=read(predictions)['predictions'];ip={r['task_id']:r for r in inferred}
    assert set(ip)==set(byid)
    saved_test_path=saved_test_predictions or fusion_dir/'predictions_fusion.json'
    saved_test={r['task_id']:r for r in read(saved_test_path)}
    max_difference=0.
    for tid,p in saved_test.items():
        actual=ip[tid];assert actual['case_decision']==p['case_decision']
        for label,value in p['case_probabilities'].items():max_difference=max(max_difference,abs(value-actual['case_probabilities'][label]))
    assert max_difference<1e-5
    comparison_file=comparison_path or fusion_dir/'comparison.json'
    comparison=read(comparison_file);gold=splits['test']
    case_targets=[r['case_decision'] for r in gold];case_predictions=[ip[r['task_id']]['case_decision'] for r in gold]
    correct=sum(a==b for a,b in zip(case_targets,case_predictions))
    assert abs(correct/len(gold)-comparison[case_key]['case']['accuracy'])<1e-12
    f1=[]
    for label in schema['fields']['case_decision']['options']:
        tp=sum(a==label and b==label for a,b in zip(case_targets,case_predictions))
        fp=sum(a!=label and b==label for a,b in zip(case_targets,case_predictions))
        fn=sum(a==label and b!=label for a,b in zip(case_targets,case_predictions))
        if tp+fn:f1.append(2*tp/(2*tp+fp+fn))
    assert abs(sum(f1)/len(f1)-comparison[case_key]['case']['macro_f1'])<1e-12
    etp=efp=efn=0
    for r in gold:
        keys=lambda es:{(e['start'],e['end'],e['label']) for e in es}
        g,p=keys(r['entities']),keys(ip[r['task_id']]['entities'])
        etp+=len(g&p);efp+=len(p-g);efn+=len(g-p)
    metrics_path=extractor_metrics or extractor_dir/'after_metrics.json'
    recorded=next(iter(read(metrics_path)['entities']['runs'].values()))['overall']['micro']
    assert (etp,efp,efn)==tuple(recorded[k] for k in ('tp','fp','fn'))
    # Exact duplicates are excluded above. This audit flags template similarity;
    # it is not a semantic duplicate detector and does not rewrite frozen splits.
    grams={tid:{r['text'][i:i+5] for i in range(max(len(r['text'])-4,0))} for tid,r in byid.items()}
    similar=[];ids=list(byid)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if real_assignment[a]==real_assignment[b] or byid[a]['patient_id']==byid[b]['patient_id']:continue
            x,y=grams[a],grams[b];score=len(x&y)/len(x|y) if x|y else 0.
            if score>=.85:similar.append(dict(task_a=a,task_b=b,split_a=real_assignment[a],split_b=real_assignment[b],character_5gram_jaccard=score))
    result=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
                frozen_files_unchanged=True,current_human_decisions_equal_frozen=True,human_cases=len(reference),point_statuses=dict(points),
                split_counts=counts,patient_count=len(owners),no_cross_split_patient_or_exact_text=True,
                synthetic_training_only=True,supervised_auxiliary_targets_recomputed_equal=True,
                numerical_vectors_in_data_not_used_by_fusion='Fusion recomputes feature vectors from predicted entities/relations; only syndrome/pathogen targets are read from data.',
                independent_cli_prediction_cases=len(inferred),test_case_correct=correct,test_case_support=len(gold),
                saved_and_cli_case_probabilities_max_difference=max_difference,strict_entity_counts=dict(tp=etp,fp=efp,fn=efn),
                near_duplicate_audit=dict(method='Cross-split different-patient character 5gram Jaccard >=0.85',flagged_pairs=similar,semantic_audit_not_performed=True),
                clinical_acceptance=False,
                input_sha256={str(p.resolve()):sha256(p) for p in [freeze_dir/'freeze_manifest.json',data_dir/'preparation_audit.json',comparison_file,metrics_path,saved_test_path,predictions]},
                verifier_sha256=sha256(Path(__file__)))
    write_json(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('freeze-dir','data-dir','extractor-dir','fusion-dir','predictions','review-dir','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();result=verify(a.freeze_dir,a.data_dir,a.extractor_dir,a.fusion_dir,a.predictions,a.review_dir,a.output)
    print(json.dumps(dict(status=result['status'],human_cases=result['human_cases'],test_correct=result['test_case_correct'],similar_pairs=len(result['near_duplicate_audit']['flagged_pairs'])),ensure_ascii=False))


if __name__=='__main__':main()
