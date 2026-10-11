"""Predict modular case decisions and extraction from task_id/text only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output path')
    import torch
    from training.modular_evaluation import modular_predict
    torch.set_num_threads(4)
    cfg=json.loads((a.checkpoint/'model_config.json').read_text('utf-8'))
    tasks=json.loads(a.tasks.read_text('utf-8-sig'));result=modular_predict(cfg,tasks)
    result={k:result[k] for k in ('predictions','reload_checks','timings','milliseconds_including_model_load','input_contract')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print('MODULAR_PREDICTED',len(result['predictions']),flush=True)


if __name__=='__main__':main()
