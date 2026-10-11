"""Verify optimization artifacts against frozen reviews and independent text-only prediction."""
import argparse
from pathlib import Path

from emr_annotation.evaluation.verify_human_hybrid_run import read,verify
from emr_annotation.training_data.schema_multitask import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('freeze-dir','data-dir','evaluation-dir','predictions','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new verification output')
    cfg=read(a.evaluation_dir/'model_config.json');receipt=read(a.evaluation_dir/'evaluation_receipt.json')
    if receipt['status']!='completed':raise ValueError('Incomplete evaluation')
    for name,digest in receipt['outputs'].items():
        if sha256(a.evaluation_dir/name)!=digest:raise ValueError(('Changed evaluation output',name))
    inferred=read(a.predictions)
    if inferred['input_contract']!='task_id_and_text_only':raise ValueError('Not text-only inference')
    cli={r['task_id']:r for r in inferred['predictions']}
    saved=read(a.evaluation_dir/'test_predictions.json')
    # Check exact entity/attribute/relation decisions as well as the case scores below.
    for row in saved:
        for key in ('text','entities','relations','case_decision','case_model'):
            if cli[row['task_id']][key]!=row[key]:raise ValueError(('Different independent prediction',row['task_id'],key))
    freeze=read(a.freeze_dir/'freeze_manifest.json')
    result=verify(a.freeze_dir,a.data_dir,Path(cfg['ner_extractor_path']),a.evaluation_dir,
                  a.predictions,Path(freeze['source']),a.output,case_key='selected',
                  extractor_metrics=a.evaluation_dir/'after_metrics.json',
                  saved_test_predictions=a.evaluation_dir/'test_predictions.json',
                  comparison_path=a.evaluation_dir/'comparison.json')
    result.update(independent_test_entities_attributes_relations_exact=True,
                  evaluation_outputs_match_receipt=True,selected_case=cfg['selected_case'],
                  selection_basis=cfg['selection'],checkpoint_reload_exact=inferred['reload_checks'],
                  modular_verifier_sha256=sha256(Path(__file__)))
    write_json(a.output,result)
    print('OPTIMIZATION_VERIFIED',result['human_cases'],result['strict_entity_counts'],result['test_case_correct'])


if __name__=='__main__':main()
