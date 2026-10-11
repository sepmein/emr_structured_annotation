"""Train context candidates, attributes and grouped case heads; lazy ML imports."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data-dir','extractor','baseline-evaluation','output-dir']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args()
    from training.error_driven_experiment import run
    run(a)


if __name__=='__main__':main()
