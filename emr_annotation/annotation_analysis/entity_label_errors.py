"""Read-only entity-label error profiles against the frozen human reference.

Strict multiset scores and deterministic descriptive error categories have
separate denominators. Only human annotations are read by this module.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess

from emr_annotation.annotation_analysis.double_annotations import entity_key
from emr_annotation.annotation_analysis.human_reference_statistics import read, overlap, score
from emr_annotation.training_data.schema_multitask import sha256, write_json


def residuals(predicted, reference):
    """Cancel exact matches one at a time; preserve duplicate predictions."""
    p_counts = Counter(entity_key(e) for e in predicted)
    g_counts = Counter(entity_key(e) for e in reference)
    common = p_counts & g_counts
    def remove(items):
        budget = common.copy()
        remaining = []
        for item in items:
            k = entity_key(item)
            if budget[k]:
                budget[k] -= 1
            else:
                remaining.append(item)
        return remaining
    return remove(predicted), remove(reference), common


def categorize(items, opposite, kind):
    """Classify each strict unmatched item once, without nearest-neighbour guessing."""
    for item in items:
        k = entity_key(item)
        same = [e for e in opposite if e['start'] == item['start'] and e['end'] == item['end']]
        hit_labels = {e['label'] for e in same}
        adjacent = [e for e in opposite if overlap(e, item)]
        if item['label'] in hit_labels:
            category = 'duplicate_prediction' if kind == 'fp' else 'reference_count_shortfall'
            target = None
        elif len(hit_labels) == 1:
            category = 'same_span_wrong_label'
            target = next(iter(hit_labels))
        elif hit_labels:
            category, target = 'ambiguous_same_span_labels', None
        elif any(e['label'] == item['label'] for e in adjacent):
            category, target = 'same_label_wrong_boundary', None
        elif adjacent:
            category, target = 'overlap_label_and_boundary', None
        else:
            category = 'extra_no_overlap' if kind == 'fp' else 'omitted_no_overlap'
            target = None
        yield item, category, target


def build(review_dir, freeze_dir, output_dir):
    reference_path = freeze_dir / 'reference.json'
    manifest = read(freeze_dir / 'freeze_manifest.json')
    for name, digest in manifest['outputs'].items():
        if sha256(freeze_dir / name) != digest:
            raise ValueError('Frozen file changed: ' + name)
    schema = read(review_dir / 'index.json')['schema']
    labels = schema['labels']
    profiles = {l:dict(label=l,pred=[],gold=[],fp=Counter(),fn=Counter(),cases=set()) for l in labels}
    flows = Counter()
    totals = Counter()
    ledger = []
    sources = {str(reference_path.resolve()): sha256(reference_path),
               str((freeze_dir/'label_config.xml').resolve()): sha256(freeze_dir/'label_config.xml'),
               str((review_dir/'index.json').resolve()): sha256(review_dir/'index.json')}
    accounts = set()
    reference = read(reference_path)
    for row in reference:
        tid = row['task_id']
        case_path = review_dir / 'cases' / (tid + '.json')
        case = read(case_path)
        sources[str(case_path.resolve())] = sha256(case_path)
        anns = case['annotations']
        if len(anns) != 2 or anns[0]['annotator'] == anns[1]['annotator']:
            raise ValueError('Expected two different people: ' + tid)
        for ann in anns:
            aid = str(ann['annotator'])
            accounts.add(aid)
            gold, pred = row['entities'], ann['entities']
            if any(e['label'] not in profiles for e in pred + gold):
                raise ValueError('Entity outside project schema: ' + tid)
            for kind, items in (('pred',pred),('gold',gold)):
                for e in items:
                    profiles[e['label']][kind].append((tid,aid,e['start'],e['end']))
                    if kind == 'gold': profiles[e['label']]['cases'].add(tid)
            p_left, g_left, common = residuals(pred,gold)
            totals.update(tp=sum(common.values()),fp=len(p_left),fn=len(g_left),annotation_views=1)
            for kind,items,opposite in (('fp',p_left,gold),('fn',g_left,pred)):
                for e,category,target in categorize(items,opposite,kind):
                    profiles[e['label']][kind][category] += 1
                    if kind == 'fp' and category == 'same_span_wrong_label':
                        flows[(e['label'],target)] += 1
                    ledger.append(dict(task_id=tid,annotator=aid,annotation_id=ann['annotation_id'],
                                       direction=kind,label=e['label'],start=e['start'],end=e['end'],
                                       category=category,opposite_label=target))
    rows = []
    for label in labels:
        p = profiles[label]
        scores = score(p['pred'],p['gold'])
        assert sum(p['fp'].values()) == scores['fp']
        assert sum(p['fn'].values()) == scores['fn']
        rows.append(dict(label=label,reference_cases=len(p['cases']),**scores,
                         fp_categories=dict(p['fp']),fn_categories=dict(p['fn']),
                         wrong_label_out=sum(c for (a,b),c in flows.items() if a == label),
                         wrong_label_in=sum(c for (a,b),c in flows.items() if b == label)))
    assert len(rows) == len(labels)
    assert sum(r['tp'] for r in rows) == totals['tp']
    assert sum(r['fp'] for r in rows) == totals['fp']
    assert sum(r['fn'] for r in rows) == totals['fn']
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    result = dict(stage='annotation_analysis',created_at=datetime.now(timezone.utc).isoformat(),
                  human_cases=len(reference),annotator_accounts=len(accounts),**dict(totals),
                  schema_labels=labels,schema_controls=schema['controls'],profiles=rows,
                  wrong_label_flows=[dict(original=a,reference=b,count=c) for (a,b),c in flows.most_common()],
                  totals_fp_categories=dict(sum((p['fp'] for p in profiles.values()),Counter())),
                  totals_fn_categories=dict(sum((p['fn'] for p in profiles.values()),Counter())),
                  scope='Only original human annotations versus human frozen reference. No AI predictions.',
                  reference_status='human_reviewed_not_expert_gold',expert_signed=False,
                  methods=dict(strict='Exact start/end/label, task and account isolated, duplicate predictions retained',
                               confusion='Original unmatched annotation with identical reference span and one unique different reference label; each original error counted once',
                               fn='Disjoint reference-side missing categories, prioritizing same span over overlapping boundaries',
                               fp='Disjoint original-side errors. Counts are not added to reference-side FN',
                               support='Each reference occurrence counted once for each of the two annotation views; cases are unique source task IDs',
                               empty_support='No gold occurrences -> recall unavailable. No gold and no prediction -> F1 unavailable',
                               group='Schema control group for descriptive charts, not person expertise rankings'),
                  source_sha256=sources,code_sha256=sha256(Path(__file__)),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  working_tree=subprocess.check_output(['git','status','--porcelain'],text=True))
    write_json(out/'entity_label_statistics.json',result)
    write_json(out/'error_ledger.json',ledger)
    with (out/'label_profiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['label','reference_cases','predicted','reference','tp','fp','fn','precision','recall','f1','wrong_label_out','wrong_label_in']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (out/'wrong_label_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['原始标签 → 裁决标签']+labels)
        for a in labels:writer.writerow([a]+[flows[(a,b)] for b in labels])
    print(json.dumps({k:result[k] for k in ('human_cases','annotator_accounts','tp','fp','fn','totals_fp_categories','totals_fn_categories')},ensure_ascii=False))
    print(json.dumps(result['wrong_label_flows'][:15],ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('review-dir','freeze-dir','output-dir'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();build(args.review_dir,args.freeze_dir,args.output_dir)


if __name__ == '__main__': main()
