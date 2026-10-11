"""Evaluate validation-selected modular mmBERT components; lazy imports."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','extractor','small','neural','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from training.modular_evaluation import run
    run(a)


if __name__=='__main__':main()
