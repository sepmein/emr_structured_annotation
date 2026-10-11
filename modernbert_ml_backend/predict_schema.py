"""Run a saved schema experiment on task_id/text inputs, without gold annotations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.resolve()==args.tasks.resolve():
        p.error('Use a new output file')
    import torch
    from transformers import AutoTokenizer
    from modeling.schema_multitask import SchemaMultitaskModel, load_schema_checkpoint
    from training.schema_experiment import predict
    cfg=json.loads((args.checkpoint/'model_config.json').read_text('utf-8'))
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SchemaMultitaskModel(cfg['base_model_path'],cfg['schema'],cfg['max_span_width'],boundary_refinement=cfg.get('boundary_refinement',False)).to(device)
    model.relation_supported_types=cfg.get('relation_supported_types', [])
    model.relation_has_negative_supervision=cfg.get('relation_has_negative_supervision',False)
    load_schema_checkpoint(model,args.checkpoint/'best_weights.pt',device,cfg['base_model_path'])
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint/'tokenizer',local_files_only=True)
    raw=json.loads(args.tasks.read_text('utf-8-sig'))
    rows=[{'task_id':r['task_id'],'text':r['text'],'entities':[]} for r in raw]
    settings=argparse.Namespace(**{k:cfg[k] for k in ('max_length','stride','max_span_width','threshold')})
    settings.span_suppression_overlap=cfg.get('span_suppression_overlap',0.)
    predictions,timings=predict(model,rows,tokenizer,settings,device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'predictions':predictions,'timings':timings,
        'reference_status':'provisional_not_medically_validated','relation_status':'positive_only_candidates_not_calibrated' if cfg.get('relation_enabled') else 'disabled_no_valid_supervision'},
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
