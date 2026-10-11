"""Fresh mmBERT/heads reconstruction; reference fields never enter prediction."""
from datetime import datetime,timezone
from pathlib import Path
import time
import joblib
import torch
from emr_annotation.evaluation.explicit_evidence_binding import bind
from emr_annotation.training_data.schema_multitask import sha256,write_json
from training.hybrid_experiment import read,prediction_inputs
from training.modular_evaluation import modular_predict
from training.error_driven_experiment import apply_candidates

def predict_selected(config,tasks):
    # Even callers passing frozen reference rows are reduced to observable inputs.
    inputs=prediction_inputs(tasks)
    path=Path(config['baseline_case_config'])
    if sha256(path)!=config['baseline_case_config_sha256']:raise ValueError('Changed case pipeline')
    baseline=read(path)
    if baseline['ner_extractor_sha256']!=config['base_extractor_sha256']:raise ValueError('Different NER extractor')
    result=modular_predict(baseline,inputs)
    rows=result['predictions'];candidate=config.get('candidate_heads')
    if candidate:
        path=Path(candidate)
        if sha256(path)!=config['candidate_heads_sha256']:raise ValueError('Changed learned candidate heads')
        heads=joblib.load(path);rows=apply_candidates(rows,heads['entity_heads'],heads['entity_choices'])
    rows=bind(rows,config['schema'],config['fields'],config['anatomical_guard'])
    return rows,result

def run(args):
    started=time.perf_counter();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);config=read(args.model_dir/'model_config.json');rows=read(args.input)
    predictions,result=predict_selected(config,rows)
    write_json(out/'predictions.json',predictions)
    # Persist actual reconstruction, not a replay of cached predictions.
    write_json(out/'reload_receipt.json',dict(status='completed',stage='evaluation',created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=sha256(args.input),config_sha256=sha256(args.model_dir/'model_config.json'),input_contract='task_id_and_exact_text_only',
        ignored_input_fields=sorted({k for r in rows for k in r if k not in ('task_id','text')}),
        checkpoint_reload_exact=result['reload_checks'],actual_fresh_neural_prediction=True,cases=len(predictions),seconds=time.perf_counter()-started,
        outputs={'predictions.json':sha256(out/'predictions.json')}))
    print('FRESH_PIPELINE_COMPLETED',len(predictions),time.perf_counter()-started,flush=True)
