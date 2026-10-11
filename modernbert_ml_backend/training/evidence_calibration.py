"""Conservative explicit evidence constraints; reference contradictions stay visible."""
from datetime import datetime,timezone
import subprocess
from pathlib import Path

from emr_annotation.evaluation.explicit_evidence_binding import bind,anatomical_conflict
from emr_annotation.evaluation.model_error_audit import inspect,render
from emr_annotation.evaluation.pathogen_attribute_audit import audit
from emr_annotation.training_data.schema_multitask import sha256,write_json
from training.hybrid_experiment import read
from training.error_driven_experiment import entity_score,apply_candidates
import joblib

ROOT=Path(__file__).resolve().parents[2]

def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    schema=read(args.data_dir/'schema.json');train=read(args.data_dir/'train.json');val=read(args.data_dir/'validation.json');pred=read(args.extractor/'validation_predictions.json')
    candidate_path=None;candidate_selection=None
    if args.candidate_heads:
        candidate_path=args.candidate_heads/'context_heads.joblib';bundle=joblib.load(candidate_path)
        proposed=apply_candidates(pred,bundle['entity_heads'],bundle['entity_choices'])
        old=entity_score(val,pred,schema)['overall']['micro'];new=entity_score(val,proposed,schema)['overall']['micro']
        candidate_selection=dict(selected=new['f1']>old['f1']+1e-9,baseline=old,candidate=new)
        if candidate_selection['selected']:pred=proposed
        else:candidate_path=None
    field_choices=[];history=[];base=audit(val,pred,schema)
    for field in ['test_result','detection_method','specimen_type']:
        candidate=bind(pred,schema,[field]);metric=audit(val,candidate,schema)
        def quality(m):
            return sum(r['known_slot_recall'] for r in m['fields'][field]['choices'] if r['support'])/sum(bool(r['support']) for r in m['fields'][field]['choices'])
        selected=quality(metric)>quality(base)+1e-9
        history.append(dict(field=field,selected=selected,baseline=base['fields'][field],candidate=metric['fields'][field]))
        if selected:field_choices.append(field)
    before=entity_score(val,pred,schema)['overall']['micro']
    filtered=entity_score(val,bind(pred,schema,guard=True),schema)['overall']['micro']
    guard=filtered['f1']>=before['f1']-1e-12
    # Accepting a non-degrading ontology guard is declared here, before any test reference is loaded.
    config=dict(format='mmbert_explicit_evidence_v1',schema=schema,base_extractor=str(args.extractor.resolve()),base_extractor_sha256=sha256(args.extractor/'best_weights.pt'),
                fields=field_choices,anatomical_guard=guard,guard_selection='Non-degrading strict validation F1; rejects explicit anatomical T/L/C/S levels from measurement labels',
                field_selection='Strict improvement in mean per-supported-value known-slot validation recall',selected_at=datetime.now(timezone.utc).isoformat())
    config.update(candidate_heads=str(candidate_path.resolve()) if candidate_path else None,candidate_heads_sha256=sha256(candidate_path) if candidate_path else None,
                  baseline_case_config=str((args.baseline_evaluation/'model_config.json').resolve()),baseline_case_config_sha256=sha256(args.baseline_evaluation/'model_config.json'))
    write_json(out/'model_config.json',config);write_json(out/'validation_candidates.json',dict(fields=history,guard=dict(selected=guard,before=before,after=filtered),candidates=candidate_selection))
    corrected_val=bind(pred,schema,field_choices,guard);write_json(out/'validation_predictions.json',corrected_val)
    write_json(out/'validation_metrics.json',dict(entities=entity_score(val,corrected_val,schema),pathogen=audit(val,corrected_val,schema)))
    questions=[dict(task_id=r['task_id'],entity=e,context=r['text'][max(0,e['start']-70):e['end']+70],status='reference_question_not_modified') for r in train+val if not r.get('synthetic') for e in r['entities'] if anatomical_conflict(r['text'],e)]
    write_json(out/'reference_questions_train_validation.json',questions)
    # Development test is evaluated only after all choices have been persisted.
    rows=read(args.data_dir/'test.json');baseline=read(args.baseline_evaluation/'extractor_predictions.json');case=read(args.baseline_evaluation/'test_predictions.json')
    if candidate_path:baseline=apply_candidates(baseline,bundle['entity_heads'],bundle['entity_choices'])
    corrected=bind(baseline,schema,field_choices,guard)
    summary,ledger,case_rows=inspect(rows,corrected,schema,train,case)
    write_json(out/'test_predictions.json',corrected);write_json(out/'test_metrics.json',dict(entities=entity_score(rows,corrected,schema),pathogen=audit(rows,corrected,schema)))
    write_json(out/'error_summary.json',summary);write_json(out/'errors.json',ledger);write_json(out/'case_comparison.json',case_rows)
    (out/'review.html').write_text(render(summary,ledger,case_rows,'文字证据约束后的开发测试'),encoding='utf-8')
    write_json(out/'reference_questions_test.json',[dict(task_id=r['task_id'],entity=e,context=r['text'][max(0,e['start']-70):e['end']+70],status='reference_question_not_modified') for r in rows for e in r['entities'] if anatomical_conflict(r['text'],e)])
    paths=[args.data_dir/(s+'.json') for s in ['train','validation','test']]+[args.data_dir/'schema.json',args.extractor/'validation_predictions.json',args.baseline_evaluation/'extractor_predictions.json',args.baseline_evaluation/'test_predictions.json']
    if candidate_path:paths.append(candidate_path)
    write_json(out/'experiment_receipt.json',dict(status='completed_development_evaluation',stage='training',schema_hash=sha256(args.data_dir/'schema.json'),
               source_sha256={str(p.resolve()):sha256(p) for p in paths},test_loaded_after_selection=True,model_weights_unchanged=True,
               method='mmBERT plus validation-selected learned entity candidates and explicit textual evidence binding; ontology guard and deterministic attributes are not newly trained neural weights',
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               code_files={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'emr_annotation/evaluation/explicit_evidence_binding.py']},
               outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file()},test_status='Test diagnostic patterns informed design; this is development evaluation, not external acceptance'))
    print('EVIDENCE_CHOICES',field_choices,'guard',guard,flush=True);print('EVIDENCE_TEST',summary['entities']['overall']['micro'],flush=True)
    print('EVIDENCE_TEST_RESULT',summary['pathogen_attributes']['fields']['test_result'],flush=True)
