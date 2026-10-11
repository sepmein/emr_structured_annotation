"""Reconstruct the validation-selected error-driven pipeline on text-only tasks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model-dir','input','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    from training.error_driven_inference import run
    run(a)
if __name__=='__main__':main()
