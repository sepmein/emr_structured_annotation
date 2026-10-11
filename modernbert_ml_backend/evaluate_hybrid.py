"""Evaluate already selected extractor/fusion snapshots once on real test cases."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','extractor','fusion','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from training.hybrid_evaluation import evaluate_selected
    evaluate_selected(a)


if __name__=='__main__':main()
