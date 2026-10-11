"""Select explicit-evidence constraints on validation, then diagnose development test."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','extractor','baseline-evaluation','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--candidate-heads',type=Path,help='Compare saved learned entity candidates on validation before test loading')
    a=p.parse_args()
    from training.evidence_calibration import run
    run(a)

if __name__=='__main__':main()
