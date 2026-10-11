"""Train separate detection-context queries, select on known real validation slots."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import random
import subprocess
import time

import torch
from torch.nn import functional as F

from modeling.attribute_context import AttributeContextHead
from training.hybrid_experiment import read,load_extractor
from training.schema_experiment import make_windows,collate
from emr_annotation.evaluation.pathogen_attribute_audit import audit
from emr_annotation.training_data.schema_multitask import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]
FIELDS=('test_result','specimen_type','detection_method')


@torch.inference_mode()
def build_cache(model,tokenizer,settings,rows,device,training=False):
    windows,_=make_windows(rows,tokenizer,model.schema,settings.max_length,settings.stride,settings.max_span_width,model.boundary_refinement)
    result=[]
    for w in windows:
        if not w['spans']:continue
        if training and not any(any(f in a for f in FIELDS) for a in w['attributes']):continue
        ids,mask,_=collate([w],device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            hidden=model.encode(ids,mask)[0];representations=model.entity_representation(hidden,w['spans'])
        result.append(dict(window=w,hidden=hidden.float().cpu(),representations=representations.float().cpu()))
    return result


@torch.inference_mode()
def apply_cache(head,cache,predictions,schema,device):
    head.eval();aggregated={}
    for pack in cache:
        w=pack['window'];spans=w['spans'];logits=head(pack['hidden'].to(device),pack['representations'].to(device),spans)
        for j,(s,e,label) in enumerate(spans):
            offsets=w['offsets'];key=(w['task_id'],offsets[s][0]+w['boundary_targets'][j][0],offsets[e-1][1]-w['boundary_targets'][j][1],schema['labels'][label])
            for f in FIELDS:
                if key[-1] not in schema['fields'][f]['scope']:continue
                slot=(*key,f)
                if slot not in aggregated:aggregated[slot]=[]
                aggregated[slot].append(logits[f][j].cpu())
    result=deepcopy(predictions)
    for row in result:
        for e in row['entities']:
            key=(row['task_id'],e['start'],e['end'],e['label'])
            for field in FIELDS:
                logits=aggregated.get((*key,field))
                if not logits:continue
                e.setdefault('attributes',{})[field]=schema['fields'][field]['options'][int(torch.stack(logits).mean(0).argmax())]
                e.setdefault('attribute_sources',{})[field]='entity_conditioned_local_attention'
    return result


def score(result):
    # Average per-value known-slot recall; majority negative cannot hide positive failures.
    per=[]
    for f in FIELDS:
        values=[r['known_slot_recall'] for r in result['fields'][f]['choices'] if r['support']]
        if values:per.append(sum(values)/len(values))
    return sum(per)/len(per)


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,tokenizer,settings,cfg=load_extractor(args.extractor,device)
    schema=cfg['schema'];model.eval()
    for p in model.parameters():p.requires_grad=False
    train=read(args.data_dir/'train.json');validation=read(args.data_dir/'validation.json');val_pred=read(args.extractor/'validation_predictions.json')
    inputs=[args.data_dir/'train.json',args.data_dir/'validation.json',args.data_dir/'schema.json',args.extractor/'best_weights.pt',args.extractor/'validation_predictions.json']
    receipt=dict(status='running',stage='training',started_at=datetime.now(timezone.utc).isoformat(),parameters=vars(args).copy(),
                 source_sha256={str(p.resolve()):sha256(p) for p in inputs},encoder_frozen=True,ner_unmodified=True,
                 selection='Mean per-supported-value known-slot recall across three detection attributes on real validation; no test selection',
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    receipt['parameters']={k:str(v) if isinstance(v,Path) else v for k,v in receipt['parameters'].items()};write_json(out/'experiment_receipt.json',receipt)
    head=AttributeContextHead(model.encoder.config.hidden_size,{f:deepcopy(model.attribute_heads[f]) for f in FIELDS},args.radius).to(device)
    train_cache=build_cache(model,tokenizer,settings,train,device,training=True)
    val_cache=build_cache(model,tokenizer,settings,val_pred,device)
    class_counts={f:Counter() for f in FIELDS}
    for row in train:
        weight=1. if not row.get('synthetic') else .05
        for e in row['entities']:
            for f in FIELDS:
                if f in e.get('attributes',{}) and e['label'] in schema['fields'][f]['scope']:class_counts[f][e['attributes'][f]]+=weight
    balancing={f:torch.tensor([min(8.,(sum(c.values())/(max(c.get(o,0),1.)*len(c)))**.5) for o in schema['fields'][f]['options']],device=device) for f,c in class_counts.items()}
    write_json(out/'training_attribute_counts.json',{f:dict(c) for f,c in class_counts.items()})
    baseline=audit(validation,val_pred,schema);best=score(baseline);best_epoch=0;best_predictions=val_pred
    torch.save(head.state_dict(),out/'best_weights.pt');history=[];optimizer=torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=.01)
    for epoch in range(1,args.epochs+1):
        head.train();losses=[];random.shuffle(train_cache)
        for pack in train_cache:
            w=pack['window'];logits=head(pack['hidden'].to(device),pack['representations'].to(device),w['spans']);terms=[]
            for f in FIELDS:
                options=schema['fields'][f]['options'];selected=[(j,options.index(a[f])) for j,a in enumerate(w['attributes']) if f in a and schema['labels'][w['spans'][j][2]] in schema['fields'][f]['scope']]
                if selected:
                    idx,y=zip(*selected);terms.append(F.cross_entropy(logits[f][list(idx)],torch.tensor(y,device=device),weight=balancing[f]))
            if not terms:continue
            loss=sum(terms)*(.05 if w.get('synthetic') else 1.);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        prediction=apply_cache(head,val_cache,val_pred,schema,device);metric=audit(validation,prediction,schema);value=score(metric)
        history.append(dict(epoch=epoch,loss=sum(losses)/max(1,len(losses)),selection_score=value,metrics=metric))
        print('ATTRIBUTE_CONTEXT_EPOCH',epoch,round(value,4),[(r['option'],r['correct'],r['support']) for r in metric['fields']['test_result']['choices']],flush=True)
        if value>best+1e-9:best=value;best_epoch=epoch;best_predictions=prediction;torch.save(head.state_dict(),out/'best_weights.pt')
    head.load_state_dict(torch.load(out/'best_weights.pt',map_location=device,weights_only=True),strict=True)
    restored=apply_cache(head,val_cache,val_pred,schema,device)
    # Epoch zero is an unchanged baseline; fresh context outputs are identically zero.
    if best_epoch and restored!=best_predictions:raise ValueError('Attribute checkpoint reload mismatch')
    if not best_epoch:restored=val_pred
    write_json(out/'validation_predictions.json',restored);write_json(out/'validation_history.json',history)
    write_json(out/'validation_metrics.json',audit(validation,restored,schema))
    config=dict(format='mmbert_attribute_context_v1',schema=schema,radius=args.radius,base_extractor=str(args.extractor.resolve()),
                base_extractor_sha256=sha256(args.extractor/'best_weights.pt'),checkpoint_sha256=sha256(out/'best_weights.pt'),best_epoch=best_epoch)
    write_json(out/'model_config.json',config);receipt.update(selected_at=datetime.now(timezone.utc).isoformat(),best_epoch=best_epoch,best_validation_score=best,checkpoint_reloaded=True)
    write_json(out/'experiment_receipt.json',receipt)
    rows=read(args.data_dir/'test.json');baseline_pred=read(args.baseline_evaluation/'extractor_predictions.json')
    test_cache=build_cache(model,tokenizer,settings,baseline_pred,device);pred=apply_cache(head,test_cache,baseline_pred,schema,device) if best_epoch else baseline_pred
    write_json(out/'test_predictions.json',pred);write_json(out/'test_metrics.json',audit(rows,pred,schema))
    receipt.update(status='completed_development_evaluation',finished_at=datetime.now(timezone.utc).isoformat(),training_seconds=time.perf_counter()-started,test_loaded_after_selection=True,
                   outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='experiment_receipt.json'},
                   code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'modernbert_ml_backend/modeling/attribute_context.py']})
    write_json(out/'experiment_receipt.json',receipt);print('ATTRIBUTE_CONTEXT_TEST',read(out/'test_metrics.json')['fields']['test_result'],flush=True)
