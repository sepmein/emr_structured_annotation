"""Double-annotation agreement and conformity to a frozen human reference.

Read-only inputs. Statistics and patient excerpts are written only to output.
Agreement between people is not correctness against the reference.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from emr_annotation.annotation_analysis.double_annotations import entity_key,kappa,metric,safe_text,relation_key
from emr_annotation.training_data.schema_multitask import sha256,write_json


def read(path):return json.loads(Path(path).read_text('utf-8'))


def score(a,g):
    a,g=Counter(a),Counter(g);tp=sum((a&g).values());fp=sum((a-g).values());fn=sum((g-a).values())
    return dict(predicted=sum(a.values()),reference=sum(g.values()),tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else None)


def overlap(a,b):return max(a['start'],b['start'])<min(a['end'],b['end'])


def normalized_annotation(a):
    entities=[{**e,'attributes':{k:v[0] for k,v in e['attributes'].items() if isinstance(v,list) and len(v)==1}} for e in a['entities']]
    case=a['case'][0]['values'][0] if len(a['case'])==1 and len(a['case'][0]['values'])==1 else None
    return entities,case


def final_relations(row):
    ids={e['id']:entity_key(e) for e in row['entities']}
    return {(ids[r['from_id']],ids[r['to_id']],r['type']) for r in row['relations']}


def original_relations(a,schema):
    values=[];invalid=0
    for r in a['relations']:
        if len(r['labels'])==len(r['source'])==len(r['target'])==1 and r['labels'][0] in schema['relations']:
            values.append((tuple(r['source'][0]),tuple(r['target'][0]),r['labels'][0]))
        else:invalid+=1
    return set(values),invalid


def excerpt(text,entity):
    s=max(0,entity['start']-45);e=min(len(text),entity['end']+45)
    return dict(start=s,end=e,text=safe_text(text[s:e]).replace('\n',' '),focus=text[entity['start']:entity['end']])


def example(tid,row,annotation,original,final,kind):
    focus=final or original
    return dict(kind=kind,task_id=tid,project=row['project'],annotator=annotation.get('annotator'),
                annotation_id=annotation.get('annotation_id'),original=original,final=final,
                excerpt=excerpt(row['text'],focus),reference_status=row['reference_status'])


def build(review_dir,freeze_dir,model_predictions,test_data,output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    freeze=read(freeze_dir/'freeze_manifest.json');reference=read(freeze_dir/'reference.json')
    for name,digest in freeze['outputs'].items():
        if sha256(freeze_dir/name)!=digest:raise ValueError('Frozen file changed: '+name)
    schema=read(review_dir/'index.json')['schema'];byid={r['task_id']:r for r in reference}
    all_a,all_b=[],[];case_pairs=[];project=defaultdict(lambda:dict(a=[],b=[],case=[],cases=0));pairs=defaultdict(lambda:dict(a=[],b=[],case=[],cases=0))
    people=defaultdict(lambda:dict(pred=[],gold=[],cases=0,case_pairs=[],attrs=Counter(),errors=Counter(),relation_hits=0,relation_support=0,relation_recorded=0,relation_invalid=0))
    attribute_pairs=defaultdict(Counter);label_pairs=defaultdict(lambda:dict(a=[],b=[]));examples=[];case_examples=[]
    audit_rows=[];both_identical=0;both_missing=0;ai_pred=[];ai_gold=[];ai_cases=[];ai_examples=[];point_statuses=Counter();source_files={}
    for tid,gold in byid.items():
        path=review_dir/'cases'/f'{tid}.json';source_files[str(path.resolve())]=sha256(path);case=read(path)
        anns=sorted(case['annotations'],key=lambda a:str(a['annotator']))
        if len(anns)!=2 or anns[0]['annotator']==anns[1]['annotator']:raise ValueError('Expected exactly two people per case')
        a,b=anns;ea,ca=normalized_annotation(a);eb,cb=normalized_annotation(b)
        ga=[(tid,*entity_key(e)) for e in ea];gb=[(tid,*entity_key(e)) for e in eb];gg=[(tid,*entity_key(e)) for e in gold['entities']]
        all_a.extend(ga);all_b.extend(gb);both_identical+=Counter(ga)==Counter(gb)
        pid=str(gold['project']);pair='/'.join(str(x['annotator']) for x in anns)
        for target in (project[pid],pairs[pair]):
            target['a'].extend(ga);target['b'].extend(gb);target['cases']+=1
            if ca is not None and cb is not None:target['case'].append((ca,cb))
        if ca is not None and cb is not None:case_pairs.append((ca,cb))
        for e in ea:label_pairs[e['label']]['a'].append((tid,e['start'],e['end']))
        for e in eb:label_pairs[e['label']]['b'].append((tid,e['start'],e['end']))
        am=defaultdict(list);bm=defaultdict(list)
        for e in ea:am[entity_key(e)].append(e)
        for e in eb:bm[entity_key(e)].append(e)
        for key in am.keys()&bm.keys():
            if len(am[key])!=1 or len(bm[key])!=1:continue
            for field,rule in schema['fields'].items():
                if not rule['per_region'] or key[2] not in rule['scope']:continue
                av,bv=am[key][0]['attributes'].get(field),bm[key][0]['attributes'].get(field)
                attribute_pairs[field]['matched_slots']+=1
                if av is None and bv is None:attribute_pairs[field]['both_blank']+=1
                elif av is None or bv is None:attribute_pairs[field]['one_blank']+=1
                elif av==bv:attribute_pairs[field]['equal_nonempty']+=1
                else:attribute_pairs[field]['different_nonempty']+=1
        final_edges=final_relations(gold)
        for annotation,entities,case_choice in ((a,ea,ca),(b,eb,cb)):
            person=people[str(annotation['annotator'])];person['cases']+=1
            person['pred'].extend((tid,*entity_key(e)) for e in entities);person['gold'].extend(gg)
            person['case_pairs'].append((case_choice,gold['case_decision']))
            rowscore=score([entity_key(e) for e in entities],[entity_key(e) for e in gold['entities']])
            audit_rows.append(dict(task_id=tid,annotator=annotation['annotator'],annotation_id=annotation['annotation_id'],**rowscore))
            mapping=defaultdict(list)
            for e in entities:mapping[entity_key(e)].append(e)
            for e in entities:
                if entity_key(e) in {entity_key(g) for g in gold['entities']}:continue
                same=[g for g in gold['entities'] if g['start']==e['start'] and g['end']==e['end']]
                adjacent=[g for g in gold['entities'] if overlap(e,g)]
                if same:
                    person['errors']['label_difference']+=1;examples.append(example(tid,gold,annotation,e,same[0],'human_label_difference'))
                elif adjacent:
                    person['errors']['boundary_or_label_overlap_difference']+=1
                else:
                    person['errors']['extra_entity_without_reference']+=1
                    examples.append(example(tid,gold,annotation,e,None,'human_extra_entity'))
            for g in gold['entities']:
                hits=mapping[entity_key(g)]
                if not hits and not any(overlap(g,e) for e in entities):
                    person['errors']['omitted_mention_no_overlap']+=1
                    examples.append(example(tid,gold,annotation,None,g,'human_omission'))
                for field,value in g.get('attributes',{}).items():
                    person['attrs']['reference_slots']+=1
                    if len(hits)==1 and hits[0]['attributes'].get(field)==value:person['attrs']['correct']+=1
                    elif len(hits)==1:
                        person['attrs']['wrong_or_blank_on_exact_entity']+=1
                        if hits[0]['attributes'].get(field) is not None:
                            item=example(tid,gold,annotation,hits[0],g,'human_attribute_difference');item.update(field=field,original_value=hits[0]['attributes'].get(field),final_value=value);examples.append(item)
                    else:person['attrs']['entity_missing_or_duplicate']+=1
            edges,invalid=original_relations(annotation,schema)
            person['relation_recorded']+=len(annotation['relations']);person['relation_invalid']+=invalid
            person['relation_support']+=len(final_edges);person['relation_hits']+=len(edges&final_edges)
            if case_choice!=gold['case_decision']:
                case_examples.append(dict(task_id=tid,annotator=annotation['annotator'],original=case_choice,final=gold['case_decision'],reason=gold.get('human_review_note') or gold.get('case_reason'),kind='human_case_difference'))
        both_missing+=sum(not any(overlap(g,e) for e in ea+eb) for g in gold['entities'])
        ai=case['sample'];ai_pred.extend((tid,*entity_key(e)) for e in ai['entities']);ai_gold.extend(gg);ai_cases.append((ai['case_decision'],gold['case_decision']))
        for e in ai['entities']:
            if entity_key(e) not in {entity_key(g) for g in gold['entities']}:
                same=[g for g in gold['entities'] if g['start']==e['start'] and g['end']==e['end']]
                item=example(tid,gold,{},e,same[0] if same else None,'ai_adjudication_draft_difference');ai_examples.append(item)
            else:
                g=next(g for g in gold['entities'] if entity_key(g)==entity_key(e))
                for field,value in g.get('attributes',{}).items():
                    if e.get('attributes',{}).get(field)!=value:
                        item=example(tid,gold,{},e,g,'ai_adjudication_attribute_difference');item.update(field=field,original_value=e.get('attributes',{}).get(field),final_value=value);ai_examples.append(item)
        point_statuses.update(p['status'] for p in gold.get('human_point_reviews',{}).values())
    person_rows=[]
    for person,p in sorted(people.items()):
        matches=sum(a==b for a,b in p['case_pairs']);attrs=p['attrs']
        person_rows.append(dict(annotator=person,cases=p['cases'],entity=score(p['pred'],p['gold']),
                                case_matches=matches,case_support=len(p['case_pairs']),case_accuracy=matches/len(p['case_pairs']),
                                attributes=dict(attrs),attribute_end_to_end_accuracy=attrs['correct']/attrs['reference_slots'] if attrs['reference_slots'] else None,
                                error_categories=dict(p['errors']),known_positive_relation_hits=p['relation_hits'],known_positive_relation_support=p['relation_support'],
                                known_positive_relation_recall=p['relation_hits']/p['relation_support'] if p['relation_support'] else None,
                                relation_recorded=p['relation_recorded'],relation_invalid_or_ambiguous=p['relation_invalid']))
    grouped=lambda groups:[dict(group=k,cases=v['cases'],entity_agreement=metric(v['a'],v['b']),case_agreement=kappa(v['case'])) for k,v in sorted(groups.items())]
    attrs=[dict(field=f,**dict(c),nonblank_equal_rate=c['equal_nonempty']/(c['matched_slots']-c['both_blank']) if c['matched_slots']>c['both_blank'] else None) for f,c in attribute_pairs.items()]
    model=read(model_predictions)['predictions'];test=read(test_data);modelmap={p['task_id']:p for p in model};model_p=[];model_g=[];model_examples=[];model_case_pairs=[]
    for gold in test:
        p=modelmap[gold['task_id']];tid=gold['task_id'];model_p.extend((tid,*entity_key(e)) for e in p['entities']);model_g.extend((tid,*entity_key(e)) for e in gold['entities']);model_case_pairs.append((p['case_decision'],gold['case_decision']))
        if p['case_decision']!=gold['case_decision']:model_examples.append(dict(kind='model_case_difference',task_id=tid,original=p['case_decision'],final=gold['case_decision'],probabilities=p['case_probabilities'],reason=gold.get('human_review_note') or gold.get('case_reason')))
        for e in p['entities']:
            if entity_key(e) not in {entity_key(g) for g in gold['entities']}:
                same=[g for g in gold['entities'] if g['start']==e['start'] and g['end']==e['end']]
                adjacent=[g for g in gold['entities'] if overlap(e,g)]
                final=(same or adjacent or [None])[0]
                model_examples.append(example(tid,gold,{},e,final,'model_false_positive_or_span_difference'))
        for g in gold['entities']:
            if not any(overlap(g,e) for e in p['entities']):model_examples.append(example(tid,gold,{},None,g,'model_omission'))
    result=dict(stage='annotation_analysis',created_at=datetime.now(timezone.utc).isoformat(),human_cases=len(reference),
                reference_status='human_reviewed_not_expert_gold',expert_signed=False,point_statuses=dict(point_statuses),
                entity_agreement=metric(all_a,all_b),case_agreement=kappa(case_pairs),case_agreement_missing_cases=len(reference)-len(case_pairs),
                identical_entity_cases=both_identical,both_people_omitted_mentions=both_missing,
                projects=grouped(project),annotator_pairs=grouped(pairs),people=person_rows,
                label_agreement=[dict(label=l,**metric(label_pairs[l]['a'],label_pairs[l]['b'])) for l in schema['labels']],
                attribute_agreement=attrs,ai_draft_vs_human=dict(entity=score(ai_pred,ai_gold),case_accuracy=sum(a==b for a,b in ai_cases)/len(ai_cases),support=len(ai_cases)),
                model_test_vs_human=dict(entity=score(model_p,model_g),case_accuracy=sum(a==b for a,b in model_case_pairs)/len(model_case_pairs),support=len(test)),
                methods=dict(entity='Strict original character start,end,label; multiset counts, task ID in key',
                             agreement_f1='2 * A/B common mentions / (A mentions + B mentions)',
                             conformity='Precision/recall/F1 against frozen human reference; duplicate mentions retained in FP counts',
                             omissions='Reference entity has no overlapping original entity; boundary and label differences listed separately',
                             case_kappa='Cohen kappa on single-choice pairs, person-pair results reported separately',
                             attributes='Agreement conditional on unique exactly matched entities; both blanks excluded from nonblank_equal_rate',
                             relations='Only known positive recall against human; missing or ambiguous original type/endpoints cannot be treated as typed matches',
                             caveats=['Human reference is current reviewed result, not independently signed expert gold','People have different case cohorts; no quality ranking','AI draft uses all182 cases; trained-model examples and metrics use27 test cases','1000 synthetic samples excluded entirely']),
                source_sha256={**source_files,str(freeze_dir/'reference.json'):sha256(freeze_dir/'reference.json'),str(model_predictions):sha256(model_predictions),str(test_data):sha256(test_data)},
                code_sha256=sha256(Path(__file__)),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                working_tree=subprocess.check_output(['git','status','--porcelain'],text=True))
    write_json(out/'statistics.json',result);write_json(out/'examples_pool.json',dict(human=examples,case=case_examples,ai_draft=ai_examples,model=model_examples));write_json(out/'per_annotation_scores.json',audit_rows)
    print(json.dumps({k:result[k] for k in ('human_cases','entity_agreement','case_agreement','people','ai_draft_vs_human','model_test_vs_human')},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('review-dir','freeze-dir','model-predictions','test-data','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();build(a.review_dir,a.freeze_dir,a.model_predictions,a.test_data,a.output_dir)


if __name__=='__main__':main()
