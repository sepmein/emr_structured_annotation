"""Train and evaluate late fusion using predicted evidence, never test gold inputs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from modeling.schema_multitask import SchemaMultitaskModel,load_schema_checkpoint
from modeling.evidence_fusion import EvidenceFusionModel
from training.schema_experiment import predict
from emr_annotation.training_data.schema_multitask import sha256,write_json
from emr_annotation.evaluation.clinical_evidence import extract,contract,SYNDROME_NAMES,PATHOGEN_NAMES,project_case_cues,CASE_CUE_NAMES

BRANCHES=('text','evidence','fusion')
ROOT=Path(__file__).resolve().parents[2]


def fusion_contract(schema,entity_context_features=False,guide_cues=False):
    spec=contract(schema)
    if guide_cues:
        spec['features'] += ['case_wording:'+name for name in CASE_CUE_NAMES]
        spec['case_wording_reference']='Project annotation guide 3.1-3.2 wording features for learned case heads; not medical truth or hard decisions'
    if entity_context_features:
        for group in ('lung','symptom','pathogen','time_measurement_exposure'):
            spec['features'] += [f'context_pool:{group}:component_{i}' for i in range(256)]+[f'context_pool:{group}:available']
        spec['context_reference']='Confidence-weighted mean of predicted entity representations; no reference entities used'
    return spec


def read(path):return json.loads(Path(path).read_text('utf-8'))


def load_extractor(directory,device):
    cfg=read(Path(directory)/'model_config.json')
    model=SchemaMultitaskModel(cfg['base_model_path'],cfg['schema'],cfg['max_span_width'],boundary_refinement=cfg.get('boundary_refinement',False)).to(device)
    model.relation_supported_types=cfg.get('relation_supported_types',[])
    model.relation_has_negative_supervision=cfg.get('relation_has_negative_supervision',False)
    model.include_document_embedding=True
    load_schema_checkpoint(model,Path(directory)/'best_weights.pt',device,cfg['base_model_path'])
    model.eval()
    tokenizer=AutoTokenizer.from_pretrained(Path(directory)/'tokenizer',local_files_only=True)
    args=SimpleNamespace(**{k:cfg[k] for k in ('max_length','stride','max_span_width','threshold')})
    args.span_suppression_overlap=cfg.get('span_suppression_overlap',0.)
    return model,tokenizer,args,cfg


def prediction_inputs(rows):
    # Ignore injected annotations, reviewer decisions and derived target fields.
    return [dict(task_id=r['task_id'],text=r['text'],entities=[]) for r in rows]


def feature_tensors(predictions,schema,device='cpu',entity_context_features=False,guide_cues=False):
    text,evidence,facts=[],[],[]
    for p in predictions:
        fact=extract(p['text'],p['entities'],p['relations'],schema)
        text.append(p['document_embedding']+p['text_case_probabilities'])
        vector=fact['vector']
        if guide_cues:vector=vector+project_case_cues(p['text'])
        if entity_context_features:
            groups=[set(schema['controls']['diagnosis_labels']),set(schema['controls']['symptons_labels']),
                    set(schema['controls']['pathogen']+schema['controls']['bacteria']+schema['controls']['other_pathogen']),
                    set(schema['controls']['time_labels']+schema['controls']['measure_entities']+schema['controls']['epidemics_labels'])]
            pooled=[]
            for group in groups:
                nodes=[e for e in p['entities'] if e['label'] in group]
                if any(len(e.get('context_embedding',[]))!=256 for e in nodes):raise ValueError('Missing predicted entity context embeddings')
                total=sum(e['score'] for e in nodes)
                pooled += [sum(e['score']*e['context_embedding'][j] for e in nodes)/total if total else 0. for j in range(256)]
                pooled.append(float(bool(nodes)))
            vector=vector+pooled
        evidence.append(vector);facts.append(fact)
    return torch.tensor(text,dtype=torch.float32,device=device),torch.tensor(evidence,dtype=torch.float32,device=device),facts


from emr_annotation.evaluation.case_classification import case_metrics


def binary_metrics(target,predicted,mask,names):
    rows=[]
    for i,name in enumerate(names):
        selected=mask[:,i].bool();a=target[selected,i].bool();b=predicted[selected,i].bool()
        tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
        rows.append(dict(label=name,known_slots=int(selected.sum()),positive_support=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
                         accuracy=(tp+tn)/int(selected.sum()) if selected.any() else None,
                         f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else None))
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return dict(known_slots=sum(r['known_slots'] for r in rows),positive_support=tp+fn,micro_f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else None,per_label=rows)


def supervision(rows,options,device,real_weight=5.,synthetic_weight=1.):
    target=[r['evidence_targets'] for r in rows]
    return dict(case=torch.tensor([options.index(r['case_decision']) for r in rows],device=device),
                syndrome=torch.tensor([r['syndromes'] for r in target],dtype=torch.float32,device=device),
                pathogen=torch.tensor([r['pathogens'] for r in target],dtype=torch.float32,device=device),
                pathogen_mask=torch.tensor([r['pathogen_mask'] for r in target],dtype=torch.float32,device=device),
                weights=torch.tensor([synthetic_weight if r.get('synthetic') else real_weight for r in rows],device=device))


def metrics(logits,targets,options):
    return dict(case=case_metrics(targets['case'].tolist(),logits['case'].argmax(-1).tolist(),options),
                syndrome=binary_metrics(targets['syndrome'],logits['syndrome'].sigmoid()>=.5,torch.ones_like(targets['syndrome']),SYNDROME_NAMES),
                pathogen=binary_metrics(targets['pathogen'],logits['pathogen'].sigmoid()>=.5,targets['pathogen_mask'],PATHOGEN_NAMES))


def loss(logits,targets,idx,class_weights):
    weights=targets['weights'][idx]
    ce=F.cross_entropy(logits['case'],targets['case'][idx],weight=class_weights,reduction='none')
    s=F.binary_cross_entropy_with_logits(logits['syndrome'],targets['syndrome'][idx],reduction='none',pos_weight=targets.get('syndrome_positive_weight')).mean(-1)
    p=F.binary_cross_entropy_with_logits(logits['pathogen'],targets['pathogen'][idx],reduction='none',pos_weight=targets.get('pathogen_positive_weight'))
    mask=targets['pathogen_mask'][idx]
    p=(p*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    return ((ce+.25*s+.25*p)*weights).sum()/weights.sum()


def serialize_predictions(rows,outputs,facts,options):
    result=[]
    for i,row in enumerate(rows):
        p=outputs['case'][i].softmax(-1).tolist();s=outputs['syndrome'][i].sigmoid().tolist();a=outputs['pathogen'][i].sigmoid().tolist()
        result.append(dict(task_id=row['task_id'],case_decision=options[int(np.argmax(p))],case_probabilities=dict(zip(options,p)),
                           syndromes=[dict(name=n,score=v,evidence_group=True) for n,v in zip(SYNDROME_NAMES,s)],
                           pathogens=[dict(name=n,score=v,structured_result_available=bool(facts[i]['pathogen_mask'][j]),status='detection_group_candidate_not_causal_diagnosis') for j,(n,v) in enumerate(zip(PATHOGEN_NAMES,a))],
                           evidence=dict(active_labels=facts[i]['active_labels'],numeric_evidence=facts[i]['numeric_evidence']),
                           reference_status='model_prediction_not_adjudicated'))
    return result


def run_fusion(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schema=read(args.data_dir/'schema.json');options=schema['fields']['case_decision']['options']
    split_names=('train','validation') if getattr(args,'validation_only',False) else ('train','validation','test')
    rows={n:read(args.data_dir/(n+'.json')) for n in split_names}
    hierarchical=getattr(args,'hierarchical_case',False)
    if hierarchical and options!=['目标病例信号','待专业复核','非目标','信息不足']:raise ValueError('Hierarchical head requires the documented case ordering')
    receipt=dict(status='running',stage='training',model='mmbert_predicted_evidence_late_fusion_v1',started_at=datetime.now(timezone.utc).isoformat(),
                 data_dir=str(args.data_dir.resolve()),extractor_dir=str(args.extractor.resolve()),device=str(device),seed=args.seed,
                 selection='Each branch selected by real validation case macro F1 only; test excluded from selection',
                 feature_policy='Both fitting and inference use only extractor predictions plus text. Gold entities never fed to fusion.',
                 auxiliary_reference='Rule-derived evidence groups on human annotations; auxiliary labels not independently human adjudicated',
                 train_feature_policy='In-sample extractor predictions for training; no out-of-fold extraction performed',
                 thresholds='Fixed 0.5 for syndrome/pathogen outputs; case argmax',
                 branch_comparison='Parallel learned heads share frozen extractor and projections; not independent encoder retraining',
                 parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/(n+'.json') for n in rows]+[args.extractor/'best_weights.pt',args.extractor/'model_config.json']},
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'modernbert_ml_backend/modeling/evidence_fusion.py',ROOT/'emr_annotation/evaluation/clinical_evidence.py']})
    write_json(out/'experiment_receipt.json',receipt)
    try:
        extractor=None;cache=getattr(args,'feature_cache',None)
        if cache:
            cached_receipt=read(cache/'experiment_receipt.json');cached_config=read(cache/'model_config.json')
            if cached_config['extractor_sha256']!=sha256(args.extractor/'best_weights.pt'):raise ValueError('Feature cache extractor mismatch')
            for n in rows:
                path=cache/('extractor_predictions_'+n+'.json')
                if sha256(path)!=cached_receipt['outputs'][path.name]:raise ValueError('Changed feature cache')
                expected=[v for p,v in cached_receipt['inputs'].items() if Path(p).name==n+'.json']
                if expected!=[sha256(args.data_dir/(n+'.json'))]:raise ValueError('Feature cache input mismatch')
            receipt['feature_cache']=dict(path=str(cache.resolve()),receipt_sha256=sha256(cache/'experiment_receipt.json'))
        else:
            extractor,tokenizer,settings,cfg=load_extractor(args.extractor,device)
            extractor.include_entity_embeddings=getattr(args,'entity_context_features',False)
        features,targets,facts,raw_predictions={}, {}, {}, {}
        for name,split in rows.items():
            print('FEATURES',name,len(split),flush=True)
            if cache:
                predictions=read(cache/('extractor_predictions_'+name+'.json'));timings=read(cache/('extractor_timings_'+name+'.json'))
                if [(p['task_id'],p['text']) for p in predictions]!=[(r['task_id'],r['text']) for r in split]:raise ValueError('Cached case/text order mismatch')
            else:predictions,timings=predict(extractor,prediction_inputs(split),tokenizer,settings,device)
            raw_predictions[name]=predictions
            x,e,facts[name]=feature_tensors(predictions,schema,device,getattr(args,'entity_context_features',False),getattr(args,'guide_cues',False))
            features[name]=(x,e);targets[name]=supervision(split,options,device,getattr(args,'real_weight',5.),getattr(args,'synthetic_weight',1.))
            write_json(out/('extractor_predictions_'+name+'.json'),predictions)
            write_json(out/('extractor_timings_'+name+'.json'),timings)
        del extractor;torch.cuda.empty_cache()
        model=EvidenceFusionModel(features['train'][0].shape[1],features['train'][1].shape[1],hidden_size=getattr(args,'hidden_size',128),hierarchical_case=hierarchical).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=getattr(args,'head_lr',.001),weight_decay=.01)
        counts=torch.bincount(targets['train']['case'],weights=targets['train']['weights'],minlength=4).float()
        class_weights=(counts.sum()/(counts.clamp_min(1)*4)).clamp(.5,3.)
        if getattr(args,'balanced_auxiliary',False):
            for key in ('syndrome','pathogen'):
                target=targets['train'][key];mask=targets['train'].get(key+'_mask',torch.ones_like(target));weight=targets['train']['weights'][:,None]*mask
                positive=(target*weight).sum(0);negative=((1-target)*weight).sum(0)
                targets['train'][key+'_positive_weight']=(negative/positive.clamp_min(1.)).sqrt().clamp(1.,4.)
        best={b:-1. for b in BRANCHES};best_epochs={};stale=0
        for epoch in range(1,args.epochs+1):
            model.train();order=torch.randperm(len(rows['train']),device=device);losses=[]
            for idx in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                outputs=model(*(v[idx] for v in features['train']))
                objective=sum(loss(outputs[b],targets['train'],idx,class_weights) for b in BRANCHES)
                if not torch.isfinite(objective):raise ValueError('Nonfinite fusion loss')
                objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(objective.detach()))
            model.eval()
            with torch.inference_mode():output=model(*features['validation'])
            report={b:metrics(output[b],targets['validation'],options) for b in BRANCHES}
            improved=False
            for branch in BRANCHES:
                score=report[branch]['case']['macro_f1']
                if score>best[branch]+.0001:
                    best[branch]=score;best_epochs[branch]=epoch;improved=True
                    torch.save(model.state_dict(),out/('best_'+branch+'.pt'))
            stale=0 if improved else stale+1
            with (out/'validation_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,loss=sum(losses)/len(losses),metrics=report),ensure_ascii=False)+'\n')
            if epoch%10==0 or epoch==1:print('FUSION_EPOCH',epoch,'loss',round(sum(losses)/len(losses),4),'validation_case_macro_f1',{b:round(report[b]['case']['macro_f1'],4) for b in BRANCHES},flush=True)
            if stale>=25:break
        comparison={};reload_checks={};evaluation_split='test' if 'test' in rows else 'validation'
        for branch in BRANCHES:
            state=torch.load(out/('best_'+branch+'.pt'),map_location=device,weights_only=True)
            if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite fusion checkpoint')
            model.load_state_dict(state,strict=True);model.eval()
            with torch.inference_mode():
                first=model(*features['validation'])[branch]['case'].clone()
                reloaded=EvidenceFusionModel(features['train'][0].shape[1],features['train'][1].shape[1],hidden_size=getattr(args,'hidden_size',128),hierarchical_case=hierarchical).to(device)
                reloaded.load_state_dict(state,strict=True);reloaded.eval()
                second=reloaded(*features['validation'])[branch]['case']
                assert torch.equal(first,second)
                test_output=reloaded(*features[evaluation_split])[branch]
            reload_checks[branch]=True
            comparison[branch]=metrics(test_output,targets[evaluation_split],options)
            write_json(out/('predictions_'+branch+'.json'),serialize_predictions(rows[evaluation_split],test_output,facts[evaluation_split],options))
        rawcase=case_metrics(targets[evaluation_split]['case'].tolist(),[options.index(p['case_decision']) for p in raw_predictions[evaluation_split]],options)
        comparison['extractor_text_case']=dict(case=rawcase)
        write_json(out/('comparison.json' if evaluation_split=='test' else 'validation_comparison.json'),comparison)
        config=dict(format='mmbert_evidence_fusion_v2' if hierarchical or getattr(args,'entity_context_features',False) else 'mmbert_evidence_fusion_v1',extractor_path=str(args.extractor.resolve()),
                    text_size=features['train'][0].shape[1],evidence_size=features['train'][1].shape[1],
                    schema=schema,evidence_contract=fusion_contract(schema,getattr(args,'entity_context_features',False),getattr(args,'guide_cues',False)),branches=list(BRANCHES),best_epochs=best_epochs,
                    checkpoint_sha256={b:sha256(out/('best_'+b+'.pt')) for b in BRANCHES},
                    extractor_sha256=sha256(args.extractor/'best_weights.pt'),selected_branch='fusion',threshold=.5)
        config.update(hierarchical_case=hierarchical,hidden_size=getattr(args,'hidden_size',128),evaluation_split=evaluation_split,
                      entity_context_features=getattr(args,'entity_context_features',False),guide_cues=getattr(args,'guide_cues',False))
        write_json(out/'model_config.json',config)
        receipt.update(status='completed' if evaluation_split=='test' else 'completed_validation_only',finished_at=datetime.now(timezone.utc).isoformat(),best_epochs=best_epochs,best_validation_case_macro_f1=best,
                       checkpoint_reload_exact=reload_checks,finite_checkpoints=True,epochs_ran=epoch,
                       comparison=comparison,outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_receipt.json'})
        write_json(out/'experiment_receipt.json',receipt);print('FUSION_COMPLETED',out,flush=True)
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));write_json(out/'experiment_receipt.json',receipt);raise


@torch.inference_mode()
def predict_hybrid(directory,tasks,device=None,branch='fusion'):
    directory=Path(directory);cfg=read(directory/'model_config.json')
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if branch not in cfg['branches']:raise ValueError('Unsupported branch')
    if sha256(Path(cfg['extractor_path'])/'best_weights.pt')!=cfg['extractor_sha256']:raise ValueError('Changed extractor checkpoint')
    if sha256(directory/('best_'+branch+'.pt'))!=cfg['checkpoint_sha256'][branch]:raise ValueError('Changed fusion checkpoint')
    started=time.perf_counter()
    extractor,tokenizer,settings,_=load_extractor(cfg['extractor_path'],device)
    extractor.include_entity_embeddings=cfg.get('entity_context_features',False)
    predictions,timings=predict(extractor,prediction_inputs(tasks),tokenizer,settings,device)
    x,e,facts=feature_tensors(predictions,cfg['schema'],device,cfg.get('entity_context_features',False),cfg.get('guide_cues',False))
    model=EvidenceFusionModel(cfg['text_size'],cfg['evidence_size'],hidden_size=cfg.get('hidden_size',128),hierarchical_case=cfg.get('hierarchical_case',False)).to(device)
    model.load_state_dict(torch.load(directory/('best_'+branch+'.pt'),map_location=device,weights_only=True),strict=True);model.eval()
    outputs=model(x,e)[branch]
    result=serialize_predictions(tasks,outputs,facts,cfg['schema']['fields']['case_decision']['options'])
    for row,p in zip(result,predictions):row.update(text=p['text'],entities=p['entities'],relations=p['relations'])
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(predictions=result,timings=timings,branch=branch,milliseconds_including_model_load=(time.perf_counter()-started)*1000,
                status='candidate_for_human_review',input_contract='task_id_and_text_only')
