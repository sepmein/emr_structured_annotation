"""CLI for provisional complete-schema experiments; no ML imports for --help."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path)
    source.add_argument('--ai-sample-reference', type=Path)
    parser.add_argument('--preserve-splits-from', type=Path)
    parser.add_argument('--label-config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--stride', type=int, default=96)
    parser.add_argument('--max-span-width', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.max_length > 512 or args.stride >= args.max_length - 2 or args.max_length < 8:
        parser.error('BERT window length must be 8..512; stride must be below content length')
    if args.batch_size < 1 or args.epochs < 1 or args.max_span_width < 1:
        parser.error('Positive batch size, epochs and max-span-width required')
    data_dir = args.output_root / 'training_data' / args.run_id
    if args.ai_sample_reference:
        if not args.preserve_splits_from:
            parser.error('--ai-sample-reference requires --preserve-splits-from')
        from emr_annotation.training_data.ai_sample import prepare
        audit = prepare(args.ai_sample_reference, args.label_config, data_dir, args.preserve_splits_from)
    else:
        from emr_annotation.training_data.schema_multitask import prepare
        audit = prepare(args.input_dir, args.label_config, data_dir, args.seed)
    print('DATA', {k: audit[k] for k in ('unique_tasks', 'annotation_views', 'split_counts', 'exclusions')}, flush=True)
    if args.check_only:
        return
    from training.schema_experiment import run
    run(args, data_dir)


if __name__ == '__main__':
    main()
