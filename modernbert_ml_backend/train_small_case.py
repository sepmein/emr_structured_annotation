"""Train validation-selected LR/XGBoost case heads; lazy ML imports."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--feature-cache',type=Path,required=True)
    p.add_argument('--text-feature-cache',type=Path,help='Optional separately verified mmBERT text branch cache')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    from training.small_case_experiment import run
    run(a)


if __name__=='__main__':main()
