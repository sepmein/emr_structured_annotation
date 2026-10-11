"""Validation-selected small case heads over text and predicted evidence.

Training reads only train/validation. Cached extractor predictions are verified
against their completed receipts and frozen inputs. No gold entity features.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from emr_annotation.evaluation.clinical_evidence import extract, contract, project_case_cues, CASE_CUE_NAMES
from emr_annotation.training_data.schema_multitask import sha256, write_json
from training.hybrid_experiment import read, case_metrics

ROOT=Path(__file__).resolve().parents[2]


def verified_cache(cache, data_dir, split):
    receipt=read(cache/'experiment_receipt.json');cfg=read(cache/'model_config.json')
    if not receipt['status'].startswith('completed'):raise ValueError('Incomplete prediction cache')
    path=cache/('extractor_predictions_'+split+'.json')
    if receipt['outputs'].get(path.name)!=sha256(path):raise ValueError('Changed prediction cache')
    inputs=[v for p,v in receipt['inputs'].items() if Path(p).name==split+'.json']
    if inputs!=[sha256(data_dir/(split+'.json'))]:raise ValueError('Changed cache source rows')
    if cfg['extractor_sha256']!=sha256(Path(cfg['extractor_path'])/'best_weights.pt'):raise ValueError('Changed extractor')
    rows=read(data_dir/(split+'.json'));pred=read(path)
    if [(p['task_id'],p['text']) for p in pred]!=[(r['task_id'],r['text']) for r in rows]:raise ValueError('Prediction identity mismatch')
    return rows,pred,cfg


def observable_features(predictions,schema):
    evidence=[];embedding=[]
    for p in predictions:
        facts=extract(p['text'],p['entities'],p['relations'],schema)
        evidence.append(facts['vector']+project_case_cues(p['text']))
        embedding.append(p['document_embedding']+p['text_case_probabilities'])
    return dict(text=[p['text'] for p in predictions],evidence=np.asarray(evidence,dtype=np.float32),
                embedding=np.asarray(embedding,dtype=np.float32))


def fit_transformer(features,kind,indices):
    prep=dict(kind=kind)
    if kind in ('text','fusion'):
        prep['tfidf']=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=8000,sublinear_tf=True)
        prep['tfidf'].fit([features['text'][i] for i in indices])
    if kind!='text':
        key='embedding' if kind=='embedding' else ('semantic_fusion' if kind=='semantic_fusion' else 'evidence')
        # Variance selection and scaling use training rows only.
        prep['columns']=np.flatnonzero(np.std(features[key][indices],axis=0)>1e-7)
        prep['scaler']=StandardScaler().fit(features[key][indices][:,prep['columns']])
    return prep


def transform(prep,features):
    kind=prep['kind'];parts=[]
    if kind in ('text','fusion'):parts.append(prep['tfidf'].transform(features['text']))
    if kind!='text':
        key='embedding' if kind=='embedding' else ('semantic_fusion' if kind=='semantic_fusion' else 'evidence')
        value=prep['scaler'].transform(features[key][:,prep['columns']]).astype(np.float32)
        parts.append(sparse.csr_matrix(value))
    return sparse.hstack(parts,format='csr') if len(parts)>1 else parts[0]


def probabilities(bundle,features,options):
    matrix=transform(bundle['preprocessor'],features)
    result=bundle['model'].predict_proba(matrix)
    full=np.zeros((len(result),len(options)),dtype=np.float64)
    for i,c in enumerate(bundle['model'].classes_):full[:,int(c)]=result[:,i]
    return full


def serial(rows,probs,options):
    return [dict(task_id=r['task_id'],case_decision=options[int(np.argmax(p))],
                 case_probabilities=dict(zip(options,map(float,p))),reference_status='model_prediction_not_adjudicated')
            for r,p in zip(rows,probs)]


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    schema=read(args.data_dir/'schema.json');options=schema['fields']['case_decision']['options']
    rows={};features={};cachecfg=None;textcfg=None
    for split in ('train','validation'):
        rows[split],pred,cachecfg=verified_cache(args.feature_cache,args.data_dir,split)
        features[split]=observable_features(pred,schema)
        if getattr(args,'text_feature_cache',None):
            textrows,textpred,textcfg=verified_cache(args.text_feature_cache,args.data_dir,split)
            if [r['task_id'] for r in textrows]!=[r['task_id'] for r in rows[split]]:raise ValueError('Text feature identity mismatch')
            features[split]['embedding']=observable_features(textpred,schema)['embedding']
        features[split]['semantic_fusion']=np.concatenate([features[split]['embedding'],features[split]['evidence']],axis=1)
    if any(r.get('synthetic') for r in rows['validation']):raise ValueError('Real validation only')
    receipt=dict(status='running',stage='training',started_at=datetime.now(timezone.utc).isoformat(),
                 model='small_case_heads_v1',selection='Real validation case macro F1; no test loaded',
                 feature_policy='Raw text plus verified predicted entities/attributes/relations and mmBERT embeddings; no reference entity inputs',
                 train_feature_policy='In-sample extractor features; no out-of-fold extraction. Raw-text features do not use extractor.',
                 data_dir=str(args.data_dir.resolve()),feature_cache=str(args.feature_cache.resolve()),
                 text_feature_cache=str(args.text_feature_cache.resolve()) if getattr(args,'text_feature_cache',None) else None,
                 inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'train.json',args.data_dir/'validation.json',args.data_dir/'schema.json',args.feature_cache/'experiment_receipt.json',args.feature_cache/'model_config.json']},
                 seed=args.seed,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'emr_annotation/evaluation/clinical_evidence.py']},
                 package_versions={name:__import__(name).__version__ for name in ('sklearn','xgboost','numpy','scipy','joblib')})
    write_json(out/'experiment_receipt.json',receipt)
    y={s:np.asarray([options.index(r['case_decision']) for r in v]) for s,v in rows.items()}
    candidates=[];best={};best_models={};allprob={};started=time.perf_counter()
    try:
        for policy in ('real_only','real_plus_low_weight_synthetic'):
            weight=np.asarray([0. if policy=='real_only' and r.get('synthetic') else (.1 if r.get('synthetic') else 1.) for r in rows['train']])
            idx=np.flatnonzero(weight>0)
            counts=np.bincount(y['train'][idx],weights=weight[idx],minlength=len(options))
            balanced=weight[idx]*counts.sum()/(len(options)*counts[y['train'][idx]])
            for kind in ('text','embedding','evidence','fusion','semantic_fusion'):
                prep=fit_transformer(features['train'],kind,idx)
                a=transform(prep,features['train'])[idx];b=transform(prep,features['validation'])
                for c in (.1,1.,10.):
                    model=LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=args.seed)
                    model.fit(a,y['train'][idx],sample_weight=balanced)
                    bundle=dict(model=model,preprocessor=prep);probs=probabilities(bundle,features['validation'],options)
                    metric=case_metrics(y['validation'].tolist(),probs.argmax(1).tolist(),options)
                    name=f'lr_{kind}_{policy}_c{c}'
                    item=dict(candidate=name,family='logistic_regression',branch=kind,policy=policy,C=c,train_rows=len(idx),validation=metric)
                    candidates.append(item);print('SMALL_CASE',name,round(metric['macro_f1'],4),flush=True)
                    if metric['macro_f1']>best.get(kind,{}).get('validation',{}).get('macro_f1',-1)+1e-8:
                        best[kind]=item;best_models[kind]=bundle;allprob[kind]=probs
            prep=fit_transformer(features['train'],'evidence',idx)
            a=transform(prep,features['train'])[idx];b=transform(prep,features['validation'])
            for depth in (2,3):
                model=XGBClassifier(n_estimators=500,max_depth=depth,learning_rate=.05,min_child_weight=3,
                                    reg_lambda=5.,reg_alpha=.5,subsample=.85,colsample_bytree=.85,
                                    objective='multi:softprob',num_class=len(options),eval_metric='mlogloss',
                                    early_stopping_rounds=30,tree_method='hist',n_jobs=4,random_state=args.seed)
                model.fit(a,y['train'][idx],sample_weight=balanced,eval_set=[(b,y['validation'])],verbose=False)
                bundle=dict(model=model,preprocessor=prep);probs=probabilities(bundle,features['validation'],options)
                metric=case_metrics(y['validation'].tolist(),probs.argmax(1).tolist(),options)
                name=f'xgb_evidence_{policy}_depth{depth}'
                item=dict(candidate=name,family='xgboost',branch='xgboost',policy=policy,depth=depth,best_iteration=model.best_iteration,
                          train_rows=len(idx),validation=metric)
                candidates.append(item);print('SMALL_CASE',name,round(metric['macro_f1'],4),flush=True)
                if metric['macro_f1']>best.get('xgboost',{}).get('validation',{}).get('macro_f1',-1)+1e-8:
                    best['xgboost']=item;best_models['xgboost']=bundle;allprob['xgboost']=probs
        selected=max(best,key=lambda k:best[k]['validation']['macro_f1'])
        hashes={};reload={}
        for branch,bundle in best_models.items():
            path=out/('best_'+branch+'.joblib');joblib.dump(bundle,path,compress=3);hashes[branch]=sha256(path)
            restored=joblib.load(path);again=probabilities(restored,features['validation'],options)
            if not np.array_equal(again,allprob[branch]):raise ValueError('Reload prediction mismatch')
            if not np.isfinite(again).all():raise ValueError('Nonfinite predictions')
            reload[branch]=True;write_json(out/('validation_predictions_'+branch+'.json'),serial(rows['validation'],again,options))
        config=dict(format='mmbert_small_case_v1',schema=schema,branches=list(best),selected_branch=selected,
                    extractor_path=cachecfg['extractor_path'],extractor_sha256=cachecfg['extractor_sha256'],
                    text_extractor_path=textcfg['extractor_path'] if textcfg else None,
                    text_extractor_sha256=textcfg['extractor_sha256'] if textcfg else None,
                    checkpoint_sha256=hashes,feature_contract=contract(schema)['features']+['case_wording:'+n for n in CASE_CUE_NAMES],
                    selection_split='validation',best_candidates=best)
        write_json(out/'model_config.json',config);write_json(out/'validation_candidates.json',candidates)
        write_json(out/'validation_comparison.json',{b:dict(case=m['validation']) for b,m in best.items()})
        receipt.update(status='completed_validation_only',finished_at=datetime.now(timezone.utc).isoformat(),
                       checkpoint_reload_exact=reload,selected_branch=selected,selected_candidate=best[selected],
                       candidate_count=len(candidates),training_seconds=time.perf_counter()-started,test_not_loaded=True,
                       outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_receipt.json'})
        write_json(out/'experiment_receipt.json',receipt);print('SMALL_CASE_COMPLETED',selected,best[selected]['validation']['macro_f1'],flush=True)
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));write_json(out/'experiment_receipt.json',receipt);raise


def load_bundle(directory,branch=None):
    cfg=read(directory/'model_config.json');branch=branch or cfg['selected_branch']
    if branch not in cfg['branches']:raise ValueError('Unsupported branch')
    path=directory/('best_'+branch+'.joblib')
    if sha256(path)!=cfg['checkpoint_sha256'][branch]:raise ValueError('Changed case model')
    return joblib.load(path),cfg,branch
