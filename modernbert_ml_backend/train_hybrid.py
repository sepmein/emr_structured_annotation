"""Train the mmBERT extractor or the text/evidence fusion stage; lazy ML imports."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    e=sub.add_parser('extractor')
    e.add_argument('--data-dir',type=Path,required=True)
    e.add_argument('--output-root',type=Path,required=True)
    e.add_argument('--run-id',required=True)
    e.add_argument('--base-model',type=Path,required=True)
    for name,default in [('epochs',12),('seed',42),('max-length',384),('stride',96),('max-span-width',32),('batch-size',4),('patience',4)]:
        e.add_argument('--'+name,type=int,default=default)
    e.add_argument('--threshold',type=float,default=.7)
    e.add_argument('--boundary-refinement',action='store_true')
    e.add_argument('--validation-only',action='store_true')
    e.add_argument('--encoder-layers',type=int,default=4)
    e.add_argument('--encoder-lr',type=float,default=2e-5)
    e.add_argument('--head-lr',type=float,default=5e-4)
    e.add_argument('--real-weight',type=float,default=1.)
    e.add_argument('--synthetic-weight',type=float,default=1.)
    e.add_argument('--negative-weight',type=float,default=1.)
    e.add_argument('--accumulation',type=int,default=1)
    e.add_argument('--warmup-fraction',type=float,default=0.)
    e.add_argument('--span-suppression-overlap',type=float,default=0.)
    e.add_argument('--selection-metric',choices=['joint','ner'],default='joint')
    e.add_argument('--initial-checkpoint',type=Path)
    e.add_argument('--validation-thresholds',default='',help='Comma-separated entity thresholds; selected on validation only')
    f=sub.add_parser('fusion')
    f.add_argument('--data-dir',type=Path,required=True)
    f.add_argument('--extractor',type=Path,required=True)
    f.add_argument('--output-dir',type=Path,required=True)
    f.add_argument('--epochs',type=int,default=160)
    f.add_argument('--seed',type=int,default=42)
    f.add_argument('--validation-only',action='store_true')
    f.add_argument('--hierarchical-case',action='store_true')
    f.add_argument('--hidden-size',type=int,default=128)
    f.add_argument('--head-lr',type=float,default=.001)
    f.add_argument('--real-weight',type=float,default=5.)
    f.add_argument('--synthetic-weight',type=float,default=1.)
    f.add_argument('--balanced-auxiliary',action='store_true')
    f.add_argument('--feature-cache',type=Path,help='Verified predictions from the exact same saved extractor and input files')
    f.add_argument('--entity-context-features',action='store_true')
    f.add_argument('--guide-cues',action='store_true',help='Project guide wording features; not new entity labels or hard case rules')
    a=p.parse_args()
    if a.stage=='extractor':
        from training.schema_experiment import run
        run(a,a.data_dir)
    else:
        from training.hybrid_experiment import run_fusion
        run_fusion(a)


if __name__=='__main__':main()
