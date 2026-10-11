"""Patient-grouped clinical focus case experiment; label-independent encoder."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,time
import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from emr_annotation.evaluation.clinical_focus import focus,features
from emr_annotation.training_data.schema_multitask import sha256,write_json
from training.hybrid_experiment import read,case_metrics
from training.error_driven_experiment import frozen_embeddings,serial_cases
ROOT=Path(__file__).resolve().parents[2]

def fit(rows,emb,indices,kind,c,balanced,options):
    vec=TfidfVectorizer(analyzer='char',ngram_range=(1,4),max_features=18000,sublinear_tf=True)
    chosen=[rows[i] for i in indices];x=vec.fit_transform([focus(r['text']) for r in chosen])
    if kind in ('focus_cues','focus_frozen_cues'):
        x=sparse.hstack([x,sparse.csr_matrix(np.asarray([features(r['text']) for r in chosen]))],format='csr')
    if kind=='focus_frozen_cues':x=sparse.hstack([x,sparse.csr_matrix(normalize(emb[indices]))],format='csr')
    model=LogisticRegression(C=c,class_weight='balanced' if balanced else None,max_iter=1500,random_state=42)
    model.fit(x,[options.index(r['case_decision']) for r in chosen])
    return dict(vectorizer=vec,model=model,kind=kind)

def probabilities(bundle,rows,emb,options):
    if bundle['kind']=='clinical_cue_trees':
        p=bundle['model'].predict_proba(np.asarray([features(r['text']) for r in rows]));full=np.zeros((len(rows),len(options)))
        for i,c in enumerate(bundle['model'].classes_):full[:,int(c)]=p[:,i]
        return full
    x=bundle['vectorizer'].transform([focus(r['text']) for r in rows]);kind=bundle['kind']
    if kind in ('focus_cues','focus_frozen_cues'):x=sparse.hstack([x,sparse.csr_matrix(np.asarray([features(r['text']) for r in rows]))],format='csr')
    if kind=='focus_frozen_cues':x=sparse.hstack([x,sparse.csr_matrix(normalize(emb))],format='csr')
    p=bundle['model'].predict_proba(x);full=np.zeros((len(rows),len(options)))
    for i,c in enumerate(bundle['model'].classes_):full[:,int(c)]=p[:,i]
    return full

def run(args):
    started=time.perf_counter();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);train=[r for r in read(args.data_dir/'train.json') if not r.get('synthetic')]
    val=read(args.data_dir/'validation.json');schema=read(args.data_dir/'schema.json');options=schema['fields']['case_decision']['options']
    cfg=read(args.extractor/'model_config.json');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    receipt=dict(status='running',stage='training',created_at=datetime.now(timezone.utc).isoformat(),seed=args.seed,
        inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'train.json',args.data_dir/'validation.json',args.data_dir/'schema.json']},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_tree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        selection_policy='Mean real validation macro F1 and patient-grouped training OOF macro F1; no test labels in features or selection',
        base_encoder='Frozen original mmBERT-base, no supervised encoder training; all upstream features label-independent',
        input_contract='task_id/text only; no entity references, case reasons, reviewed metadata or patient identifiers used as features',
        test_status='Previously used development diagnostic batch, not external acceptance')
    write_json(out/'experiment_receipt.json',receipt)
    if args.trees:
        te=np.zeros((len(train),1));ve=np.zeros((len(val),1))
        receipt['base_encoder']='Not used: clinical text cues only'
    else:
        focus_rows=[dict(task_id=r['task_id'],text=focus(r['text'])) for r in train+val]
        emb=frozen_embeddings(focus_rows,cfg['base_model_path'],device);te,ve=emb[:len(train)],emb[len(train):]
        np.savez_compressed(out/'frozen_focus_train_validation_embeddings.npz',train=te,validation=ve)
    y=np.asarray([options.index(r['case_decision']) for r in train]);groups=np.asarray([str(r['patient_id']) for r in train]);indices=np.arange(len(train))
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=args.seed).split(indices,y,groups))
    fold_data=[dict(fold=i,train_task_ids=[train[j]['task_id'] for j in a],heldout_task_ids=[train[j]['task_id'] for j in b],patient_overlap=sorted(set(groups[a])&set(groups[b]))) for i,(a,b) in enumerate(folds)]
    assert not any(r['patient_overlap'] for r in fold_data);write_json(out/'grouped_case_folds.json',fold_data)
    history=[];best=-1
    for kind in (['clinical_cue_trees'] if args.trees else ['focus','focus_cues','focus_frozen_cues']):
        for c in ([3,6,None] if args.trees else [.3,3.]):
            for balanced in [False,True]:
                oof=np.zeros((len(train),len(options)))
                def train_bundle(a):
                    if not args.trees:return fit(train,te,a,kind,c,balanced,options)
                    model=ExtraTreesClassifier(n_estimators=200,max_depth=c,min_samples_leaf=3,class_weight='balanced' if balanced else None,random_state=args.seed,n_jobs=1)
                    model.fit(np.asarray([features(train[i]['text']) for i in a]),y[a])
                    return dict(kind=kind,model=model)
                for a,b in folds:oof[b]=probabilities(train_bundle(a),[train[i] for i in b],te[b],options)
                om=case_metrics(y.tolist(),oof.argmax(1).tolist(),options)
                bundle=train_bundle(indices);p=probabilities(bundle,val,ve,options)
                vm=case_metrics([options.index(r['case_decision']) for r in val],p.argmax(1).tolist(),options);score=(om['macro_f1']+vm['macro_f1'])/2
                row=dict(kind=kind,C=c,balanced=balanced,oof=om,validation=vm,selection_score=score);history.append(row)
                print('FOCUS_CASE',kind,c,balanced,round(om['macro_f1'],4),round(vm['macro_f1'],4),flush=True)
                if score>best+1e-9:best=score;selected=row;chosen=bundle;vp=p;selected_oof=oof
    joblib.dump(chosen,out/'case_head.joblib',compress=3);restored=joblib.load(out/'case_head.joblib')
    assert np.array_equal(probabilities(restored,val,ve,options),vp)
    write_json(out/'validation_candidates.json',history);write_json(out/'validation_case_predictions.json',serial_cases(val,vp,options))
    write_json(out/'oof_case_predictions.json',serial_cases(train,selected_oof,options))
    config=dict(format='clinical_focus_case_v1',schema=schema,selected=selected,base_model_path=cfg['base_model_path'],checkpoint_sha256=sha256(out/'case_head.joblib'),selected_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'model_config.json',config)
    # Selected artifact is persisted before the development test is loaded.
    test=read(args.data_dir/'test.json')
    if args.trees:fe=np.zeros((len(test),1))
    else:
        fe=frozen_embeddings([dict(task_id=r['task_id'],text=focus(r['text'])) for r in test],cfg['base_model_path'],device)
        np.save(out/'frozen_focus_test_embeddings.npy',fe)
    tp=probabilities(restored,test,fe,options)
    tm=case_metrics([options.index(r['case_decision']) for r in test],tp.argmax(1).tolist(),options)
    write_json(out/'test_case_predictions.json',serial_cases(test,tp,options));write_json(out/'test_metrics.json',tm)
    receipt.update(status='completed_development_evaluation',selected=selected,checkpoint_reload_exact=True,test_reference_loaded_after_selection=True,test_input_sha256=sha256(args.data_dir/'test.json'),training_seconds=time.perf_counter()-started,
        code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'emr_annotation/evaluation/clinical_focus.py']},outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_receipt.json'})
    write_json(out/'experiment_receipt.json',receipt);print('FOCUS_CASE_TEST',tm,flush=True)
