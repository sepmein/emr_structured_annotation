"""Train patient-grouped text-only clinical focus heads, then evaluate development."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data-dir','extractor','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',action='store_true',help='Bounded nonlinear clinical-cue comparison, no encoder loading')
    a=p.parse_args()
    from training.focused_case_experiment import run
    run(a)
if __name__=='__main__':main()
