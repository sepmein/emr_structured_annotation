"""Error-driven context heads. Fit/select before loading development test references."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel,AutoTokenizer

from emr_annotation.evaluation.context_candidates import PATTERNS,candidates,context,annotation_questions
from emr_annotation.evaluation.entity_predictions import normalize_records,score_run,entity_metrics
from emr_annotation.evaluation.pathogen_attribute_audit import audit as attribute_audit
from emr_annotation.evaluation.clinical_evidence import project_case_cues
from emr_annotation.evaluation.model_error_audit import inspect,render
from emr_annotation.training_data.schema_multitask import sha256,write_json
from training.hybrid_experiment import read,case_metrics

ROOT=Path(__file__).resolve().parents[2]


def entity_score(rows,pred,schema):
    labels=set(schema['labels'])
    return score_run(normalize_records(rows,labels,reference=True,source='reference'),normalize_records(pred,labels,reference=False,source='pred'),{'schema':schema['labels']})


def fit_local(examples,targets,weights,c):
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,4),min_df=1,max_features=18000,sublinear_tf=True)
    x=vectorizer.fit_transform(examples)
    model=LogisticRegression(C=c,max_iter=1000,solver='lbfgs',random_state=42)
    model.fit(x,targets,sample_weight=weights)
    return dict(vectorizer=vectorizer,model=model)


def local_prob(bundle,examples):
    return bundle['model'].predict_proba(bundle['vectorizer'].transform(examples))


def train_candidates(train,validation,predictions,schema):
    heads={};choices={};history=[]
    real=[r for r in train if not r.get('synthetic')]
    for label in PATTERNS:
        mention_counts=Counter(r['text'][e['start']:e['end']] for r in real for e in r['entities'] if e['label']==label)
        extra={label:sorted(m for m,n in mention_counts.items() if n>=2 and not any(c.isdigit() for c in m) and len(m)<=15)}
        examples=[];targets=[]
        for r in real:
            gold={(e['start'],e['end'],e['label']) for e in r['entities']}
            # Standard supervised positive spans; only text proposals are used at inference.
            proposals=candidates(r['text'],r['entities'],extra)
            for e in proposals:
                if e['label']!=label:continue
                examples.append(context(r['text'],e));targets.append(int((e['start'],e['end'],label) in gold))
        y=np.asarray(targets);frequency=np.bincount(y,minlength=2)
        if min(frequency)==0:continue
        weights=np.asarray([frequency.sum()/(2*frequency[v]) for v in y])
        baseline=entity_score(validation,predictions,schema)['per_label'][label]
        best=baseline['f1'];choice=dict(mode='unchanged',validation_f1=best,extra_mentions=extra)
        for c in (.3,3.):
            bundle=fit_local(examples,y,weights,c)
            for mode in ['filter','complete']:
                for threshold in [.35,.5,.65,.8]:
                    candidate=dict(mode=mode,threshold=threshold,extra_mentions=extra)
                    trial=apply_candidates(predictions,{label:bundle},{label:candidate})
                    metric=entity_score(validation,trial,schema)['per_label'][label]
                    record=dict(label=label,C=c,**candidate,validation=metric)
                    history.append(record)
                    if metric['f1'] is not None and (best is None or metric['f1']>best+1e-9):
                        best=metric['f1'];choice=record;heads[label]=bundle
        choices[label]=choice
        if choice['mode']=='unchanged':heads.pop(label,None)
        print('CANDIDATE_HEAD',label,choice['mode'],best,flush=True)
    return heads,choices,history


def apply_candidates(predictions,heads,choices):
    output=deepcopy(predictions)
    for row in output:
        old={(e['start'],e['end'],e['label']):e for e in row['entities']}
        kept=[e for e in row['entities'] if e['label'] not in heads]
        for label,bundle in heads.items():
            choice=choices[label]
            proposals=[e for e in candidates(row['text'],row['entities'],choice['extra_mentions']) if e['label']==label]
            if not proposals:continue
            p=local_prob(bundle,[context(row['text'],e) for e in proposals])[:,list(bundle['model'].classes_).index(1)]
            for e,prob in zip(proposals,p):
                key=(e['start'],e['end'],label)
                if prob<choice['threshold'] or (choice['mode']=='filter' and key not in old):continue
                kept.append(deepcopy(old[key]) if key in old else dict(**e,attributes={},score=float(prob),candidate_origin='learned_character_context_completion'))
        row['entities']=sorted(kept,key=lambda e:(e['start'],e['end'],e['label']))
        for i,e in enumerate(row['entities']):e.setdefault('id',f'context:{i}')
        ids={e['id'] for e in row['entities']}
        row['relations']=[r for r in row.get('relations',[]) if r['from_id'] in ids and r['to_id'] in ids]
    return output


def train_attributes(train,validation,predictions,schema):
    heads={};choices={};history=[]
    by_id={r['task_id']:r for r in predictions}
    for field,config in schema['fields'].items():
        if not config['per_region']:continue
        examples=[];targets=[];weights=[]
        for r in train:
            for e in r['entities']:
                value=e.get('attributes',{}).get(field)
                if value is not None and e['label'] in config['scope']:
                    examples.append((r['text'],e));targets.append(value);weights.append(.1 if r.get('synthetic') else 1.)
        frequency=Counter(targets)
        if len(frequency)<2:continue
        gold=[];observed=[];original=[]
        for r in validation:
            entities={(e['start'],e['end'],e['label']):e for e in by_id[r['task_id']]['entities']}
            for e in r['entities']:
                target=e.get('attributes',{}).get(field);p=entities.get((e['start'],e['end'],e['label']))
                if target is not None and e['label'] in config['scope'] and p:
                    gold.append(target);observed.append((r['text'],p));original.append(p.get('attributes',{}).get(field))
        if not observed:continue
        # All declared options remain in the macro denominator for this field.
        baseline=case_metrics([config['options'].index(v) for v in gold],[config['options'].index(v) if v in config['options'] else -1 for v in original],config['options'])
        best=baseline['macro_f1'];choice=dict(mode='unchanged',validation=baseline)
        for radius in (12,40):
            texts=[context(t,e,radius) for t,e in examples]
            val=[context(t,e,radius) for t,e in observed]
            for c in (.3,3.):
                for balancing in ('plain','balanced'):
                    total=sum(weights);count=Counter()
                    for v,w in zip(targets,weights):count[v]+=w
                    fit_weights=weights if balancing=='plain' else [w*total/(len(count)*count[v]) for w,v in zip(weights,targets)]
                    bundle=fit_local(texts,targets,fit_weights,c);probs=local_prob(bundle,val)
                    answer=bundle['model'].classes_[probs.argmax(1)]
                    metric=case_metrics([config['options'].index(v) for v in gold],[config['options'].index(v) for v in answer],config['options'])
                    item=dict(field=field,radius=radius,C=c,balancing=balancing,validation=metric)
                    history.append(item)
                    if metric['macro_f1']>best+1e-9:best=metric['macro_f1'];choice=item;heads[field]=bundle
        choices[field]=choice
        print('ATTRIBUTE_HEAD',field,choice.get('mode','learned_context'),best,flush=True)
    return heads,choices,history


def apply_attributes(predictions,heads,choices,schema):
    output=deepcopy(predictions)
    for row in output:
        for field,bundle in heads.items():
            ents=[e for e in row['entities'] if e['label'] in schema['fields'][field]['scope']]
            if not ents:continue
            probs=local_prob(bundle,[context(row['text'],e,choices[field]['radius']) for e in ents])
            for e,prob in zip(ents,probs):
                e.setdefault('attributes',{})[field]=str(bundle['model'].classes_[prob.argmax()])
                e.setdefault('attribute_sources',{})[field]='entity_conditioned_character_context_head'
    return output


@torch.inference_mode()
def frozen_embeddings(rows,base_path,device):
    tokenizer=AutoTokenizer.from_pretrained(base_path,local_files_only=True)
    model=AutoModel.from_pretrained(base_path,local_files_only=True).to(device).eval();cache={};vectors=[]
    for i,row in enumerate(rows):
        text=row['text']
        if text not in cache:
            enc=tokenizer(text,max_length=512,truncation=True,stride=64,return_overflowing_tokens=True,return_tensors=None)
            total=None;count=0
            for ids in enc['input_ids']:
                x=torch.tensor([ids],device=device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    hidden=model(input_ids=x,attention_mask=torch.ones_like(x)).last_hidden_state[0,1:-1].float()
                total=hidden.sum(0) if total is None else total+hidden.sum(0);count+=len(hidden)
            cache[text]=(total/max(count,1)).cpu().numpy()
        vectors.append(cache[text])
        if i%40==0:print('FROZEN_EMBED',i,len(rows),flush=True)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return np.asarray(vectors)


def fit_case(rows,embeddings,indices,kind,c,balanced,options):
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,4),min_df=1,max_features=18000,sublinear_tf=True)
    x=vectorizer.fit_transform([rows[i]['text'] for i in indices])
    if kind=='text_frozen':x=sparse.hstack([x,sparse.csr_matrix(normalize(embeddings[indices]))],format='csr')
    model=LogisticRegression(C=c,class_weight='balanced' if balanced else None,max_iter=1500,random_state=42)
    model.fit(x,[options.index(rows[i]['case_decision']) for i in indices])
    return dict(vectorizer=vectorizer,model=model,kind=kind)


def case_prob(bundle,rows,embeddings,options):
    x=bundle['vectorizer'].transform([r['text'] for r in rows])
    if bundle['kind']=='text_frozen':x=sparse.hstack([x,sparse.csr_matrix(normalize(embeddings))],format='csr')
    p=bundle['model'].predict_proba(x);full=np.zeros((len(rows),len(options)))
    for j,c in enumerate(bundle['model'].classes_):full[:,c]=p[:,j]
    return full


def serial_cases(rows,p,options):
    return [dict(task_id=r['task_id'],case_decision=options[int(v.argmax())],case_probabilities=dict(zip(options,map(float,v)))) for r,v in zip(rows,p)]


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4)
    train=read(args.data_dir/'train.json');validation=read(args.data_dir/'validation.json');schema=read(args.data_dir/'schema.json')
    cfg=read(args.extractor/'model_config.json');base_val=read(args.extractor/'validation_predictions.json')
    if any(r.get('synthetic') for r in validation):raise ValueError('Real validation only')
    exrec=read(args.extractor/'experiment_receipt.json')
    if sha256(args.extractor/'best_weights.pt')!=exrec['outputs']['best_weights.pt']:raise ValueError('Changed extractor')
    if sha256(args.data_dir/'validation.json')!=exrec['inputs']['validation.json']:raise ValueError('Changed validation reference')
    if exrec['outputs']['validation_predictions.json']!=sha256(args.extractor/'validation_predictions.json'):raise ValueError('Changed validation predictions')
    receipt=dict(status='running',stage='training',created_at=datetime.now(timezone.utc).isoformat(),seed=args.seed,
                 inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'train.json',args.data_dir/'validation.json',args.data_dir/'schema.json',args.extractor/'experiment_receipt.json',args.extractor/'validation_predictions.json']},
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 working_tree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                 selection_policy='Entity and attribute heads real validation only; case mean grouped-training OOF and real validation macro F1. Development test not used for selection.',
                 pretrained_embedding_policy='Frozen original mmBERT-base encoder; no train/validation case labels used by encoder. Grouped OOF case fitting is label-independent upstream.',
                 test_status='Previously used development batch; this run uses its diagnostic error patterns. Not external acceptance.')
    write_json(out/'experiment_receipt.json',receipt)
    write_json(out/'reference_questions.json',annotation_questions([r for r in train if not r.get('synthetic')]+validation))
    eheads,echoices,ehistory=train_candidates(train,validation,base_val,schema)
    aheads,achoices,ahistory=train_attributes(train,validation,base_val,schema)
    corrected_val=apply_attributes(apply_candidates(base_val,eheads,echoices),aheads,achoices,schema)
    joblib.dump(dict(entity_heads=eheads,entity_choices=echoices,attribute_heads=aheads,attribute_choices=achoices),out/'context_heads.joblib',compress=3)
    real=[r for r in train if not r.get('synthetic')];options=schema['fields']['case_decision']['options']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb=frozen_embeddings(real+validation,cfg['base_model_path'],device)
    train_emb,val_emb=emb[:len(real)],emb[len(real):];np.savez_compressed(out/'frozen_train_validation_embeddings.npz',train=train_emb,validation=val_emb)
    y=np.asarray([options.index(r['case_decision']) for r in real]);groups=np.asarray([str(r['patient_id']) for r in real]);indices=np.arange(len(real))
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=args.seed).split(indices,y,groups))
    fold_rows=[dict(fold=i,train_task_ids=[real[j]['task_id'] for j in a],heldout_task_ids=[real[j]['task_id'] for j in b],patient_overlap=sorted(set(groups[a])&set(groups[b]))) for i,(a,b) in enumerate(folds)]
    if any(r['patient_overlap'] for r in fold_rows):raise ValueError('Patient fold overlap')
    write_json(out/'grouped_case_folds.json',fold_rows)
    case_history=[];best=-1;best_bundle=None;best_prob=None;selection=None
    for kind in ['text','text_frozen']:
        for c in [.3,3.]:
            for balanced in [False,True]:
                oof=np.zeros((len(real),len(options)))
                for a,b in folds:
                    bundle=fit_case(real,train_emb,a,kind,c,balanced,options)
                    oof[b]=case_prob(bundle,[real[j] for j in b],train_emb[b],options)
                oof_metric=case_metrics(y.tolist(),oof.argmax(1).tolist(),options)
                bundle=fit_case(real,train_emb,indices,kind,c,balanced,options);p=case_prob(bundle,validation,val_emb,options)
                metric=case_metrics([options.index(r['case_decision']) for r in validation],p.argmax(1).tolist(),options)
                score=(oof_metric['macro_f1']+metric['macro_f1'])/2
                item=dict(kind=kind,C=c,balanced=balanced,oof=oof_metric,validation=metric,selection_score=score)
                case_history.append(item);print('GROUPED_CASE',kind,c,balanced,round(oof_metric['macro_f1'],4),round(metric['macro_f1'],4),flush=True)
                if score>best+1e-9:best=score;best_bundle=bundle;best_prob=p;selection=item
    joblib.dump(best_bundle,out/'case_head.joblib',compress=3)
    restored=joblib.load(out/'case_head.joblib');again=case_prob(restored,validation,val_emb,options)
    if not np.array_equal(again,best_prob):raise ValueError('Case reload mismatch')
    restored_heads=joblib.load(out/'context_heads.joblib')
    check=apply_attributes(apply_candidates(base_val,restored_heads['entity_heads'],restored_heads['entity_choices']),restored_heads['attribute_heads'],restored_heads['attribute_choices'],schema)
    if check!=corrected_val:raise ValueError('Context reload mismatch')
    write_json(out/'validation_predictions.json',corrected_val);write_json(out/'validation_case_predictions.json',serial_cases(validation,best_prob,options))
    write_json(out/'validation_metrics.json',dict(entities=entity_score(validation,corrected_val,schema),pathogen=attribute_audit(validation,corrected_val,schema),case=selection['validation']))
    write_json(out/'validation_candidates.json',dict(entities=ehistory,attributes=ahistory,cases=case_history))
    config=dict(format='mmbert_context_hybrid_v1',schema=schema,base_extractor=str(args.extractor.resolve()),base_extractor_sha256=sha256(args.extractor/'best_weights.pt'),base_model_path=cfg['base_model_path'],
                entity_choices=echoices,attribute_choices=achoices,selected_case=selection,case_embedding='Frozen mmBERT non-special token mean across 512-token overlapping windows',
                checkpoint_sha256={p.name:sha256(p) for p in [out/'case_head.joblib',out/'context_heads.joblib']},input_contract='task_id_and_exact_text_only')
    # Persist immutable decisions before loading any development test reference.
    write_json(out/'model_config.json',config)
    receipt.update(status='completed_validation_only',selected_at=datetime.now(timezone.utc).isoformat(),checkpoint_reload_exact=True,test_not_loaded=True,training_seconds=time.perf_counter()-started)
    write_json(out/'experiment_receipt.json',receipt)
    if args.evaluate_test:
        rows=read(args.data_dir/'test.json');baseline=read(args.baseline_evaluation/'extractor_predictions.json')
        corrected=apply_attributes(apply_candidates(baseline,eheads,echoices),aheads,achoices,schema)
        test_emb=frozen_embeddings(rows,cfg['base_model_path'],device);np.save(out/'frozen_test_embeddings.npy',test_emb)
        p=case_prob(best_bundle,rows,test_emb,options);cp=serial_cases(rows,p,options)
        summary,ledger,case_rows=inspect(rows,corrected,schema,train,cp)
        write_json(out/'test_predictions.json',corrected);write_json(out/'test_case_predictions.json',cp)
        write_json(out/'test_metrics.json',dict(entities=entity_score(rows,corrected,schema),pathogen=attribute_audit(rows,corrected,schema),case=case_metrics([options.index(r['case_decision']) for r in rows],p.argmax(1).tolist(),options)))
        write_json(out/'error_summary.json',summary);write_json(out/'errors.json',ledger);write_json(out/'case_comparison.json',case_rows)
        (out/'review.html').write_text(render(summary,ledger,case_rows,'改进后开发测试'),encoding='utf-8')
        receipt.update(status='completed_development_evaluation',test_not_loaded=False,test_reference_loaded_after_selection=True,
                       test_inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'test.json',args.baseline_evaluation/'extractor_predictions.json']})
        print('TEST_ENTITY',summary['entities']['overall']['micro'],flush=True);print('TEST_CASE',read(out/'test_metrics.json')['case'],flush=True)
    receipt.update(finished_at=datetime.now(timezone.utc).isoformat(),outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_receipt.json'},
                   code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'emr_annotation/evaluation/context_candidates.py',ROOT/'emr_annotation/evaluation/model_error_audit.py']})
    write_json(out/'experiment_receipt.json',receipt);print('ERROR_DRIVEN_COMPLETED',out,flush=True)
