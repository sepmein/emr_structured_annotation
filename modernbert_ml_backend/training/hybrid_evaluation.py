"""Actual selected-model evaluation, kept separate from validation-only training."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess

import torch

from training.hybrid_experiment import (read,load_extractor,prediction_inputs,feature_tensors,
    supervision,metrics,serialize_predictions,BRANCHES,case_metrics)
from training.schema_experiment import predict,evaluate
from modeling.evidence_fusion import EvidenceFusionModel
from emr_annotation.training_data.schema_multitask import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def evaluate_selected(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=read(args.fusion/'model_config.json');rows=read(args.data_dir/'test.json');options=cfg['schema']['fields']['case_decision']['options']
    if any(r.get('synthetic') for r in rows):raise ValueError('Real test only')
    if cfg['extractor_sha256']!=sha256(args.extractor/'best_weights.pt'):raise ValueError('Changed selected extractor')
    ereceipt=read(args.extractor/'experiment_receipt.json');freceipt=read(args.fusion/'experiment_receipt.json')
    if not ereceipt.get('checkpoint_reloaded') or not all(freceipt['checkpoint_reload_exact'].values()):raise ValueError('Missing reload evidence')
    receipt=dict(status='running',stage='evaluation',started_at=datetime.now(timezone.utc).isoformat(),
                 data_dir=str(args.data_dir.resolve()),extractor_path=str(args.extractor.resolve()),fusion_path=str(args.fusion.resolve()),
                 selection='Frozen validation-selected checkpoints and thresholds; no test tuning in this evaluator',
                 reference_status='human_reviewed_not_expert_gold',
                 test_batch_status='Previously examined development batch; not an untouched external acceptance set',
                 inputs={str(p.resolve()):sha256(p) for p in [args.data_dir/'test.json',args.extractor/'best_weights.pt',args.extractor/'model_config.json',args.fusion/'model_config.json']},
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'modernbert_ml_backend/training/schema_experiment.py',ROOT/'modernbert_ml_backend/training/hybrid_experiment.py',ROOT/'modernbert_ml_backend/modeling/evidence_fusion.py']})
    write_json(out/'evaluation_receipt.json',receipt)
    try:
        model,tokenizer,settings,_=load_extractor(args.extractor,device)
        model.include_entity_embeddings=cfg.get('entity_context_features',False)
        predictions,timings=predict(model,prediction_inputs(rows),tokenizer,settings,device)
        # This separate call only diagnoses attribute errors conditional on
        # known reference entities; its results never enter the fusion inputs.
        conditional,_=predict(model,rows,tokenizer,settings,device,gold_attributes=True)
        write_json(out/'extractor_predictions.json',predictions);write_json(out/'extractor_timings.json',timings)
        write_json(out/'after_metrics.json',evaluate(rows,predictions,cfg['schema'],conditional))
        x,e,facts=feature_tensors(predictions,cfg['schema'],device,cfg.get('entity_context_features',False),cfg.get('guide_cues',False))
        del model;torch.cuda.empty_cache()
        targets=supervision(rows,options,device)
        comparison={};reload={}
        for branch in BRANCHES:
            path=args.fusion/('best_'+branch+'.pt')
            if sha256(path)!=cfg['checkpoint_sha256'][branch]:raise ValueError('Changed selected fusion checkpoint')
            state=torch.load(path,map_location=device,weights_only=True)
            if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite fusion checkpoint')
            network=EvidenceFusionModel(cfg['text_size'],cfg['evidence_size'],hidden_size=cfg.get('hidden_size',128),hierarchical_case=cfg.get('hierarchical_case',False)).to(device)
            network.load_state_dict(state,strict=True);network.eval()
            with torch.inference_mode():logits=network(x,e)[branch]
            network.load_state_dict(state,strict=True)
            with torch.inference_mode():second=network(x,e)[branch]
            if not all(torch.equal(logits[k],second[k]) for k in logits):raise ValueError('Non-exact checkpoint reconstruction')
            comparison[branch]=metrics(logits,targets,options);reload[branch]=True
            write_json(out/('predictions_'+branch+'.json'),serialize_predictions(rows,logits,facts,options))
        comparison['extractor_text_case']=dict(case=case_metrics(targets['case'].tolist(),[options.index(p['case_decision']) for p in predictions],options))
        write_json(out/'comparison.json',comparison)
        receipt.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),test_cases=len(rows),
                       checkpoint_reload_exact=reload,finite_checkpoints=True,entity_threshold=settings.threshold,
                       outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='evaluation_receipt.json'})
        write_json(out/'evaluation_receipt.json',receipt);print('EVALUATION_COMPLETED',out,flush=True)
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));write_json(out/'evaluation_receipt.json',receipt);raise
