"""Fit entity-conditioned attribute attention without changing the NER encoder."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data-dir','extractor','baseline-evaluation','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--radius',type=int,default=40);p.add_argument('--learning-rate',type=float,default=.001)
    a=p.parse_args()
    from training.attribute_context_experiment import run
    run(a)

if __name__=='__main__':main()
