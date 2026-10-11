"""Explicit GPU/artifact acceptance checks for an AI sample experiment."""
import argparse
import json
from pathlib import Path
import sys

from emr_annotation.training_data.schema_multitask import write_json, sha256
from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.annotation_analysis.double_annotations import parse_annotation


def verify(training_dir, data_dir, adjudication_dir, output):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'modernbert_ml_backend'))
    import torch
    from transformers import AutoTokenizer
    from modeling.schema_multitask import SchemaMultitaskModel
    from training.schema_experiment import predict, make_windows
    read=lambda p:json.loads(Path(p).read_text('utf-8'))
    cfg=read(training_dir/'model_config.json');rec=read(training_dir/'experiment_receipt.json')
    schema=load_double_annotation_schema(data_dir/'label_config.xml')
    tasks=read(adjudication_dir/'label_studio_ai_samples.json')
    parse_issues=[i for t in tasks for i in parse_annotation(t,t['predictions'][0],schema,'AI','sample')['issues']]
    assert not parse_issues, 'Label Studio normalized export has structural issues'
    summaries=read(adjudication_dir/'point_summary.json')
    ledger=read(adjudication_dir/'point_decisions.json')
    raw=[p for p in ledger if 'source_annotation_id' in p]
    assert len(raw)==summaries['original_results']==len({p['id'] for p in raw})
    for filename,digest in rec['outputs'].items():
        assert sha256(training_dir/filename)==digest
    for filename,digest in rec['inputs'].items():
        assert sha256(data_dir/filename)==digest
    samples=read(adjudication_dir/'proposals.json')
    assert len(samples)==182 and all(s['case_reviewed_by_ai'] and not s['expert_signed'] for s in samples)
    source_samples={s['task_id']:s for s in samples}
    for name in ('train','validation','test'):
        assert all(source_samples[r['task_id']]['text']==r['text'] for r in read(data_dir/(name+'.json')))
    torch.set_num_threads(4);torch.manual_seed(rec['parameters']['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SchemaMultitaskModel(cfg['base_model_path'],cfg['schema'],cfg['max_span_width']).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.relation_head.state_dict().items()}
    weights=torch.load(training_dir/'best_weights.pt',map_location=device,weights_only=True)
    assert all(torch.isfinite(v).all() for v in weights.values())
    model.load_state_dict(weights,strict=True);model.relation_supported_types=cfg['relation_supported_types']
    changed=[k for k,v in model.relation_head.state_dict().items() if not torch.equal(initial[k],v.detach().cpu())]
    assert changed
    tokenizer=AutoTokenizer.from_pretrained(training_dir/'tokenizer',local_files_only=True)
    settings=argparse.Namespace(**{k:cfg[k] for k in ('max_length','stride','max_span_width','threshold')})
    plain={'task_id':'synthetic','text':'虚构样例。无发热。体温37℃。','entities':[]}
    marked={**plain,'entities':[{'id':'irrelevant','start':0,'end':2,'label':'肺炎诊断','attributes':{}}],
            'relations':[],'case_decision':'目标病例信号'}
    a,_=predict(model,[plain],tokenizer,settings,device);b,_=predict(model,[marked],tokenizer,settings,device)
    assert a==b
    long={'task_id':'synthetic_long','text':'虚构正常记录。'*180+'末尾发热。','entities':[]}
    windows,_=make_windows([long],tokenizer,cfg['schema'],settings.max_length,settings.stride,settings.max_span_width)
    assert max(e for w in windows for _,e in w['offsets'])==len(long['text'])
    labeled={'task_id':'synthetic_pairs','text':'发热3天，体温38℃。','entities':[
        {'id':'a','start':0,'end':2,'label':'发热','attributes':{}},
        {'id':'b','start':2,'end':4,'label':'时间表达','attributes':{}},
        {'id':'c','start':5,'end':7,'label':'体温','attributes':{}}],
        'relations':[{'from_id':'a','to_id':'b','type':'持续时长'}],'relations_supervised':True}
    w,_=make_windows([labeled],tokenizer,cfg['schema'],384,96,32)
    assert sum(len(x['pair_targets']) for x in w)==1
    result={'status':'passed_local_sample_acceptance','checkpoint_reload':True,'all_weights_finite':True,
            'input_and_output_hashes_verified':True,'relation_head_changed_tensors':changed,
            'inference_independent_of_reference_answers':True,'full_text_last_character_covered':True,
            'unknown_relation_pairs_not_negative':True,'original_results_all_unique_traced':len(raw),
            'normalized_LS_prediction_parse_issues':len(parse_issues),'LS_live_UI_test':False,
            'privacy_gate_deferred_by_user':True,'medical_expert_acceptance':False,
            'code_sha256':sha256(__file__),'device':str(device)}
    write_json(output,result);print(json.dumps(result,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-dir','data-dir','adjudication-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Use a new verification output file')
    verify(a.training_dir,a.data_dir,a.adjudication_dir,a.output)


if __name__=='__main__':
    main()
