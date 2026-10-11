"""Predict cases, evidence groups and detection groups from task_id/text only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--branch',choices=['text','evidence','fusion'],default='fusion')
    a=p.parse_args()
    if a.output.exists() or a.output.resolve()==a.tasks.resolve():p.error('Use a new output path')
    import torch
    from training.hybrid_experiment import predict_hybrid
    torch.set_num_threads(4)
    tasks=json.loads(a.tasks.read_text('utf-8-sig'))
    result=predict_hybrid(a.checkpoint,tasks,branch=a.branch)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print('PREDICTED',len(result['predictions']),flush=True)


if __name__=='__main__':main()
