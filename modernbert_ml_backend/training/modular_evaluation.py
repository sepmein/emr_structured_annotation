"""Select on validation, evaluate and serve task-specific mmBERT components."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from modeling.evidence_fusion import EvidenceFusionModel
from training.hybrid_experiment import (read,load_extractor,prediction_inputs,feature_tensors,
    supervision,metrics,serialize_predictions,case_metrics,BRANCHES)
from training.schema_experiment import predict,evaluate
from training.small_case_experiment import observable_features,load_bundle,probabilities,serial
from emr_annotation.training_data.schema_multitask import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def selected_configuration(args):
    nc=read(args.neural/'model_config.json');nr=read(args.neural/'experiment_receipt.json')
    sc=read(args.small/'model_config.json');sr=read(args.small/'experiment_receipt.json')
    er=read(args.extractor/'experiment_receipt.json')
    if not er.get('checkpoint_reloaded') or not all(nr['checkpoint_reload_exact'].values()) or not all(sr['checkpoint_reload_exact'].values()):raise ValueError('Unverified selected checkpoint')
    nb=max(nr['best_validation_case_macro_f1'],key=nr['best_validation_case_macro_f1'].get)
    sb=sc['selected_branch'];ns=nr['best_validation_case_macro_f1'][nb];ss=sc['best_candidates'][sb]['validation']['macro_f1']
    family,branch,score=('small',sb,ss) if ss>ns else ('neural',nb,ns)
    return dict(format='mmbert_modular_v1',schema=sc['schema'],ner_extractor_path=str(args.extractor.resolve()),
                ner_extractor_sha256=sha256(args.extractor/'best_weights.pt'),
                small_path=str(args.small.resolve()),small_config_sha256=sha256(args.small/'model_config.json'),
                neural_path=str(args.neural.resolve()),neural_config_sha256=sha256(args.neural/'model_config.json'),
                selected_case=dict(family=family,branch=branch,validation_macro_f1=score),auxiliary_neural_branch=nb,
                feature_policy='All prediction paths use task_id/text only; task-specific extractors may differ',
                selection='All case family/branch choices determined by real validation macro F1 before test prediction')


@torch.inference_mode()
def modular_predict(cfg,tasks,device=None,all_branches=False,diagnostic_rows=None):
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    started=time.perf_counter();options=cfg['schema']['fields']['case_decision']['options'];timings={};cache={}
    if sha256(Path(cfg['ner_extractor_path'])/'best_weights.pt')!=cfg['ner_extractor_sha256']:raise ValueError('Changed NER extractor')
    if sha256(Path(cfg['small_path'])/'model_config.json')!=cfg['small_config_sha256'] or sha256(Path(cfg['neural_path'])/'model_config.json')!=cfg['neural_config_sha256']:raise ValueError('Changed modular configuration')
    nc=read(Path(cfg['neural_path'])/'model_config.json');sc=read(Path(cfg['small_path'])/'model_config.json')
    expected={cfg['ner_extractor_path']:cfg['ner_extractor_sha256'],nc['extractor_path']:nc['extractor_sha256'],sc['extractor_path']:sc['extractor_sha256']}
    if sc.get('text_extractor_path'):expected[sc['text_extractor_path']]=sc['text_extractor_sha256']
    conditional=None
    for path,digest in expected.items():
        if sha256(Path(path)/'best_weights.pt')!=digest:raise ValueError('Changed task extractor')
        model,tokenizer,settings,_=load_extractor(path,device)
        model.include_entity_embeddings=nc.get('entity_context_features',False) and path==nc['extractor_path']
        preds,ts=predict(model,prediction_inputs(tasks),tokenizer,settings,device)
        cache[path]=preds;timings[path]=ts
        if path==cfg['ner_extractor_path'] and diagnostic_rows is not None:
            conditional,_=predict(model,diagnostic_rows,tokenizer,settings,device,gold_attributes=True)
        del model;torch.cuda.empty_cache()
    x,e,facts=feature_tensors(cache[nc['extractor_path']],cfg['schema'],device,nc.get('entity_context_features',False),nc.get('guide_cues',False))
    nb=cfg['auxiliary_neural_branch'];neural_branches=list(BRANCHES) if all_branches else [nb]
    outputs={};reload={}
    for branch in neural_branches:
        path=Path(cfg['neural_path'])/('best_'+branch+'.pt')
        if sha256(path)!=nc['checkpoint_sha256'][branch]:raise ValueError('Changed neural head')
        state=torch.load(path,map_location=device,weights_only=True)
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite neural weights')
        network=EvidenceFusionModel(nc['text_size'],nc['evidence_size'],hidden_size=nc.get('hidden_size',128),hierarchical_case=nc.get('hierarchical_case',False)).to(device)
        network.load_state_dict(state,strict=True);network.eval();logits=network(x,e)[branch]
        network.load_state_dict(state,strict=True);again=network(x,e)[branch]
        if not all(torch.equal(logits[k],again[k]) for k in logits):raise ValueError('Neural reconstruction mismatch')
        outputs['neural_'+branch]=logits;reload['neural_'+branch]=True
    features=observable_features(cache[sc['extractor_path']],cfg['schema'])
    if sc.get('text_extractor_path'):features['embedding']=observable_features(cache[sc['text_extractor_path']],cfg['schema'])['embedding']
    features['semantic_fusion']=np.concatenate([features['embedding'],features['evidence']],axis=1)
    small_branches=sc['branches'] if all_branches else ([cfg['selected_case']['branch']] if cfg['selected_case']['family']=='small' else [])
    small_probs={}
    for branch in small_branches:
        bundle,_,_=load_bundle(Path(cfg['small_path']),branch)
        p=probabilities(bundle,features,options);second=probabilities(bundle,features,options)
        if not np.isfinite(p).all() or not np.array_equal(p,second):raise ValueError('Invalid small-head probabilities')
        small_probs['small_'+branch]=p;reload['small_'+branch]=True
    result=serialize_predictions(tasks,outputs['neural_'+nb],facts,options)
    key=cfg['selected_case']['family']+'_'+cfg['selected_case']['branch']
    if cfg['selected_case']['family']=='small':
        selected=serial(tasks,small_probs[key],options)
        for r,c in zip(result,selected):r.update(case_decision=c['case_decision'],case_probabilities=c['case_probabilities'])
    for r,p in zip(result,cache[cfg['ner_extractor_path']]):
        r.update(text=p['text'],entities=p['entities'],relations=p['relations'],case_model=key)
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(predictions=result,extractor_predictions=cache[cfg['ner_extractor_path']],conditional=conditional,
                neural_outputs=outputs,neural_facts=facts,small_probabilities=small_probs,reload_checks=reload,timings=timings,
                milliseconds_including_model_load=(time.perf_counter()-started)*1000,input_contract='task_id_and_text_only')


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    # Selection is completed and recorded before the test file is read.
    cfg=selected_configuration(args);write_json(out/'model_config.json',cfg)
    receipt=dict(status='running',stage='evaluation',started_at=datetime.now(timezone.utc).isoformat(),
                 selected_case=cfg['selected_case'],selection=cfg['selection'],
                 test_batch_status='Previously examined development batch; not untouched external acceptance',
                 inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'test.json',args.extractor/'experiment_receipt.json',args.small/'experiment_receipt.json',args.neural/'experiment_receipt.json']},
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'modernbert_ml_backend/training/small_case_experiment.py',ROOT/'modernbert_ml_backend/training/schema_experiment.py',ROOT/'emr_annotation/evaluation/clinical_evidence.py']})
    write_json(out/'evaluation_receipt.json',receipt)
    try:
        rows=read(args.data_dir/'test.json')
        if any(r.get('synthetic') for r in rows):raise ValueError('Real test only')
        outputs=modular_predict(cfg,prediction_inputs(rows),all_branches=True,diagnostic_rows=rows)
        options=cfg['schema']['fields']['case_decision']['options'];target=supervision(rows,options,'cpu')
        comparison={}
        for branch,logits in outputs['neural_outputs'].items():
            comparison[branch]=metrics({k:v.cpu() for k,v in logits.items()},target,options)
            write_json(out/('predictions_'+branch+'.json'),serialize_predictions(rows,{k:v.cpu() for k,v in logits.items()},outputs['neural_facts'],options))
        for branch,p in outputs['small_probabilities'].items():
            comparison[branch]=dict(case=case_metrics(target['case'].tolist(),p.argmax(1).tolist(),options))
            write_json(out/('predictions_'+branch+'.json'),serial(rows,p,options))
        selected=cfg['selected_case']['family']+'_'+cfg['selected_case']['branch'];comparison['selected']=comparison[selected]
        write_json(out/'comparison.json',comparison)
        write_json(out/'after_metrics.json',evaluate(rows,outputs['extractor_predictions'],cfg['schema'],outputs['conditional']))
        write_json(out/'extractor_predictions.json',outputs['extractor_predictions']);write_json(out/'test_predictions.json',outputs['predictions']);write_json(out/'timings.json',outputs['timings'])
        receipt.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),test_cases=len(rows),
                       checkpoint_reload_exact=outputs['reload_checks'],finite_predictions=True,
                       input_contract=outputs['input_contract'],milliseconds_including_model_load=outputs['milliseconds_including_model_load'],
                       outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='evaluation_receipt.json'})
        write_json(out/'evaluation_receipt.json',receipt);print('MODULAR_EVALUATION_COMPLETED',selected,comparison['selected']['case']['macro_f1'],flush=True)
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));write_json(out/'evaluation_receipt.json',receipt);raise
