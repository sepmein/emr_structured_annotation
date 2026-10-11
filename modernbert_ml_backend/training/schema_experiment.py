"""Reproducible provisional multitask training and strict held-out evaluation."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modeling.schema_multitask import (SchemaMultitaskModel, span_loss, unfreeze_encoder_tail,
                                      save_schema_checkpoint, load_schema_checkpoint)
from emr_annotation.training_data.schema_multitask import sha256, write_json, relation_type_allowed
from emr_annotation.evaluation.ner_re import relation_keys
from emr_annotation.evaluation.entity_predictions import build_report, normalize_records, entity_metrics
from emr_annotation.training_data.span_alignment import align_entity,suppress_redundant_spans

ROOT = Path(__file__).resolve().parents[2]


def make_windows(rows, tokenizer, schema, length, stride, width, refine_boundaries=False):
    windows, audit = [], Counter()
    label_ids = {v: i for i, v in enumerate(schema['labels'])}
    case_options = schema['fields']['case_decision']['options']
    for row in rows:
        encoded = tokenizer(row['text'], max_length=length, truncation=True, stride=stride,
                            return_overflowing_tokens=True, return_offsets_mapping=True)
        covered = set()
        count = len(encoded['input_ids'])
        audit['documents'] += 1
        audit['windows'] += count
        for input_ids, offsets in zip(encoded['input_ids'], encoded['offset_mapping']):
            starts = {s: i for i, (s, e) in enumerate(offsets) if e > s}
            ends = {e: i + 1 for i, (s, e) in enumerate(offsets) if e > s}
            spans, attributes, indexes, boundary_targets, ignored_regions = [], [], {}, [], []
            lo = min(starts, default=0)
            hi = max(ends, default=0)
            safe_ner = row.get('ner_supervised', True)
            for entity in row['entities']:
                s, e = entity['start'], entity['end']
                aligned=align_entity(offsets,s,e,width,refine_boundaries)
                if aligned:
                    begin,finish,left,right=aligned
                    span = (begin,finish,label_ids[entity['label']])
                    indexes[entity['id']] = len(spans)
                    spans.append(span)
                    attributes.append(entity.get('attributes', {}))
                    boundary_targets.append((left,right))
                    audit['refined_boundary_targets'] += int(bool(left or right))
                    covered.add(entity['id'])
                elif s < hi and e > lo:
                    # Partial/unrepresentable positive mentions must not become negatives.
                    if refine_boundaries:
                        ignored_regions.append((s,e,label_ids[entity['label']]))
                    else:safe_ner = False
            pairs, pair_targets = [], []
            if row.get('relations_supervised'):
                observed = {}
                for rel in row['relations']:
                    observed.setdefault((rel['from_id'], rel['to_id']), set()).add(rel['type'])
                for (a, b), types in observed.items():
                    # Unknown pairs and conflicting multi-type pairs have no loss.
                    if a in indexes and b in indexes and a != b and len(types) == 1:
                        pairs.append((indexes[a], indexes[b]))
                        pair_targets.append(schema['relations'].index(next(iter(types))) + 1)
                        audit['positive_relation_window_targets'] += 1
                if row.get('relations_complete'):
                    # Only explicitly complete synthetic references provide
                    # no-edge targets. Unobserved real pairs remain unknown.
                    linked = set(pairs)
                    for a, (_, _, source_label) in enumerate(spans):
                        for b, (_, _, target_label) in enumerate(spans):
                            if a != b and (a,b) not in linked and any(relation_type_allowed(schema['labels'][source_label],schema['labels'][target_label],t,schema) for t in schema['relations']):
                                pairs.append((a,b));pair_targets.append(0)
                                audit['synthetic_negative_relation_window_targets'] += 1
            case = row.get('case_decision')
            windows.append({'input_ids': input_ids, 'offsets': offsets, 'spans': spans,
                            'attributes': attributes, 'ner_supervised': safe_ner,
                            'case_target': case_options.index(case) if case in case_options else -100,
                            'case_weight': 1 / count, 'task_id': row['task_id'],
                            'sample_weight':row.get('sample_weight',1.),
                            'synthetic':row.get('synthetic',False),'boundary_targets':boundary_targets,'ignored_regions':ignored_regions,
                            'pairs': pairs, 'pair_targets': pair_targets})
            audit['ner_supervised_windows'] += int(safe_ner)
        audit['entities'] += len(row['entities'])
        audit['entities_covered_by_any_window'] += len(covered)
        audit['unrepresentable_entities'] += len(row['entities']) - len(covered)
    return windows, dict(audit)


def collate(batch, device):
    length = max(len(w['input_ids']) for w in batch)
    ids = torch.zeros((len(batch), length), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, window in enumerate(batch):
        n = len(window['input_ids'])
        ids[i, :n] = torch.tensor(window['input_ids'], device=device)
        mask[i, :n] = 1
    return ids, mask, batch


def training_loss(model, hidden, logits, batch):
    losses = []
    n = len(model.schema['labels'])
    for i, window in enumerate(batch):
        sample_losses=[]
        length = len(window['input_ids'])
        local = logits[i, :length]
        if window['ner_supervised']:
            valid = torch.zeros_like(local, dtype=torch.bool)
            real = torch.tensor([e > s for s, e in window['offsets']], device=local.device)
            for w in range(model.max_span_width):
                k = length - w
                if k > 0:
                    valid[:k, w] = (real[:k] & real[w:]).unsqueeze(-1)
            target = torch.zeros_like(valid)
            for s, e, label in window['spans']:
                target[s, e - s - 1, label] = True
            # Partial or excessively long reference spans must not become
            # negative candidates; preserve supervision elsewhere in the window.
            for s,e,label in window.get('ignored_regions',[]):
                for w in range(model.max_span_width):
                    k=length-w
                    if k>0:
                        overlap=torch.tensor([window['offsets'][j][0]<e and window['offsets'][j+w][1]>s for j in range(k)],device=local.device)
                        valid[:k,w,label]&=~overlap
            sample_losses.append(span_loss(local,target,valid,getattr(model,'negative_weight',1.)))
        representations = model.entity_representation(hidden[i], window['spans'])
        for field, head in model.attribute_heads.items():
            options = model.schema['fields'][field]['options']
            selected = [(j, options.index(a[field])) for j, a in enumerate(window['attributes']) if field in a]
            if selected:
                indexes, targets = zip(*selected)
                sample_losses.append(getattr(model,'attribute_loss_weight',.15) * F.cross_entropy(head(representations[list(indexes)]),
                              torch.tensor(targets, device=hidden.device)))
        if getattr(model,'boundary_refinement',False) and len(representations):
            for side,head in enumerate(model.boundary_heads):
                weights=hidden.new_ones(model.max_boundary_trim+1);weights[0]=.4
                sample_losses.append(.25*F.cross_entropy(head(representations),torch.tensor([v[side] for v in window['boundary_targets']],device=hidden.device),weight=weights))
        if window['case_target'] != -100:
            case_logits = model.case_head(hidden[i, :1])
            sample_losses.append(getattr(model,'case_loss_weight',.5) * window['case_weight'] * F.cross_entropy(case_logits,
                          torch.tensor([window['case_target']], device=hidden.device)))
        if window['pairs']:
            rel = model.relation_logits(hidden[i], representations, window['pairs'], window['spans'])
            sample_losses.append(getattr(model,'relation_loss_weight',.5) * F.cross_entropy(rel, torch.tensor(window['pair_targets'], device=hidden.device)))
        losses.append(sum(sample_losses)*window.get('sample_weight',1.) if sample_losses else logits[i].sum()*0)
    return sum(losses) / len(batch) if losses else logits.sum() * 0


@torch.inference_mode()
def predict(model, rows, tokenizer, args, device, gold_attributes=False):
    model.eval()
    results, timings, cache = [], [], {}
    for row in rows:
        text = row['text']
        if text in cache and not gold_attributes:
            pred = deepcopy(cache[text])
            pred['task_id'] = row['task_id']
            results.append(pred)
            continue
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        windows, _ = make_windows([row], tokenizer, model.schema, args.max_length, args.stride, args.max_span_width,getattr(model,'boundary_refinement',False))
        entities, cases, relations, document_embeddings = {}, [], {}, []
        for window in windows:
            ids, mask, _ = collate([window], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                hidden = model.encode(ids, mask)
                cases.append(model.case_head(hidden[:, 0]).float().squeeze(0))
                if getattr(model, 'include_document_embedding', False):
                    document_embeddings.append(hidden[0,0].float())
                scores = model.span_logits(hidden).sigmoid()[0]
                if gold_attributes:
                    spans = window['spans']
                    probabilities = [1.] * len(spans)
                else:
                    candidates = (scores >= args.threshold).nonzero().tolist()
                    spans, probabilities = [], []
                    for begin, w, label in candidates:
                        end = begin + w + 1
                        offsets = window['offsets']
                        if (end <= len(offsets) and offsets[begin][1] > offsets[begin][0]
                                and offsets[end-1][1] > offsets[end-1][0]):
                            spans.append((begin, end, label))
                            probabilities.append(float(scores[begin, w, label]))
                representations = model.entity_representation(hidden[0], spans)
                offsets_by_span=[(window['offsets'][begin][0],window['offsets'][end-1][1]) for begin,end,_ in spans]
                if getattr(model,'boundary_refinement',False) and spans:
                    trims=[head(representations).argmax(-1).tolist() for head in model.boundary_heads]
                    offsets_by_span=[(s+trims[0][j],e-trims[1][j]) for j,(s,e) in enumerate(offsets_by_span)] if not gold_attributes else [(e['start'],e['end']) for e in row['entities'] if align_entity(window['offsets'],e['start'],e['end'],args.max_span_width,True)]
                attribute_values = {f: head(representations).argmax(-1).tolist()
                                    for f, head in model.attribute_heads.items()} if spans else {}
                supported = getattr(model, 'relation_supported_types', [])
                pair_candidates, allowed_types = [], []
                if supported:
                    for a, (_, _, source_label) in enumerate(spans):
                        for b, (_, _, target_label) in enumerate(spans):
                            possible = [name for name in supported if a != b and
                                        relation_type_allowed(model.schema['labels'][source_label],
                                                              model.schema['labels'][target_label], name, model.schema)]
                            if possible:
                                pair_candidates.append((a, b))
                                allowed_types.append(possible)
                if pair_candidates:
                    rel_logits = model.relation_logits(hidden[0], representations, pair_candidates, spans).float()
                    for j, ((a, b), possible) in enumerate(zip(pair_candidates, allowed_types)):
                        use_no_edge = getattr(model, 'relation_has_negative_supervision', False)
                        indexes = ([0] if use_no_edge else [])+[model.schema['relations'].index(name) + 1 for name in possible]
                        scores_type = rel_logits[j, indexes].softmax(-1)
                        choice = int(scores_type.argmax())
                        confidence = float(scores_type[choice])
                        if use_no_edge and (choice == 0 or float(scores_type[choice]) < .5):
                            continue
                        if use_no_edge:
                            choice -= 1
                        source, target = spans[a], spans[b]
                        def char_key(span):
                            begin, end, label = span
                            j=spans.index(span);s,e=offsets_by_span[j]
                            return (s,e,model.schema['labels'][label])
                        key = (char_key(source), char_key(target), possible[choice])
                        relations[key] = max(relations.get(key, 0), confidence)
            for j, ((begin, end, label), probability) in enumerate(zip(spans, probabilities)):
                s,e=offsets_by_span[j]
                if not 0<=s<e<=len(text):continue
                name = model.schema['labels'][label]
                attrs = {f: model.schema['fields'][f]['options'][vals[j]]
                         for f, vals in attribute_values.items() if name in model.schema['fields'][f]['scope']}
                key = (s, e, name)
                entity = {'start': s, 'end': e, 'label': name, 'score': probability, 'attributes': attrs}
                if getattr(model,'include_entity_embeddings',False):entity['context_embedding']=representations[j].float().cpu().tolist()
                if key not in entities or entities[key]['score'] < probability:
                    entities[key] = entity
        values = suppress_redundant_spans(list(entities.values()),getattr(args,'span_suppression_overlap',0.))
        for i, entity in enumerate(values):
            entity['id'] = f'e{i}'
        id_by_key = {(e['start'],e['end'],e['label']):e['id'] for e in values}
        relation_values = [{'from_id':id_by_key[source], 'to_id':id_by_key[target], 'type':name,
                            'direction':'right','score':confidence,
                            'status':('candidate_synthetic_negative_supervision_not_clinically_calibrated'
                                      if getattr(model,'relation_has_negative_supervision',False) else
                                      'candidate_positive_only_supervision_not_calibrated_edge_presence')}
                           for (source,target,name),confidence in relations.items() if source in id_by_key and target in id_by_key]
        prediction = {'task_id': row['task_id'], 'text': text, 'entities': values, 'relations': relation_values,
                      'case_decision': model.schema['fields']['case_decision']['options'][int(torch.stack(cases).mean(0).argmax())]}
        if document_embeddings:
            prediction['document_embedding'] = torch.stack(document_embeddings).mean(0).cpu().tolist()
            prediction['text_case_probabilities'] = torch.stack(cases).mean(0).softmax(-1).cpu().tolist()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings.append({'task_id': row['task_id'], 'characters': len(text), 'windows': len(windows),
                        'milliseconds': (time.perf_counter() - started) * 1000,
                        'measurement': 'local end-to-end one document, tokenization+GPU+decode; no service/network'})
        cache[text] = deepcopy(prediction)
        results.append(prediction)
    return results, timings


def evaluate(rows, predictions, schema, conditional=None):
    labels = set(schema['labels'])
    included = {r['task_id'] for r in rows if r.get('ner_supervised', True)}
    gold = normalize_records([r for r in rows if r['task_id'] in included], labels, reference=True, source='provisional')
    pred = normalize_records([r for r in predictions if r['task_id'] in included], labels, reference=False, source='predicted')
    entity_report = build_report(gold, pred, schema['controls'])
    predmap = {r['task_id']: r for r in predictions}
    condmap = {r['task_id']: r for r in conditional or []}
    attrs, cases = {}, []
    for field, rule in schema['fields'].items():
        if not rule['per_region']:
            continue
        support = correct = conditional_correct = 0
        distribution = Counter()
        for row in rows:
            index = {(e['start'], e['end'], e['label']): e for e in predmap[row['task_id']]['entities']}
            cindex = {(e['start'], e['end'], e['label']): e for e in condmap.get(row['task_id'], {}).get('entities', [])}
            for ent in row['entities']:
                if field not in ent.get('attributes', {}):
                    continue
                support += 1
                target = ent['attributes'][field]
                distribution[target] += 1
                key = (ent['start'], ent['end'], ent['label'])
                correct += index.get(key, {}).get('attributes', {}).get(field) == target
                conditional_correct += cindex.get(key, {}).get('attributes', {}).get(field) == target
        attrs[field] = {'support': support, 'correct': correct,
                        'end_to_end_known_slot_accuracy': correct / support if support else None,
                        'gold_entity_correct': conditional_correct if conditional is not None else None,
                        'gold_entity_known_slot_accuracy': conditional_correct / support if support and conditional is not None else None,
                        'reference_options': dict(distribution)}
    for row in rows:
        if row.get('case_decision'):
            cases.append((row['case_decision'], predmap[row['task_id']]['case_decision']))
    per_class = {}
    for option in schema['fields']['case_decision']['options']:
        tp = sum(a == option and b == option for a, b in cases)
        fp = sum(a != option and b == option for a, b in cases)
        fn = sum(a == option and b != option for a, b in cases)
        per_class[option] = entity_metrics(tp, fp, fn)
    supported = [m['f1'] or 0 for m in per_class.values() if m['support']]
    relation_support = relation_hits = conditional_hits = 0
    relation_distribution = Counter()
    for row in rows:
        known = Counter(relation_keys(row, schema['relations'], reference=True))
        candidate = Counter(relation_keys(predmap[row['task_id']], schema['relations'], reference=False))
        oracle = Counter(relation_keys(condmap.get(row['task_id'], {'entities':[],'relations':[]}), schema['relations'], reference=False))
        relation_support += sum(known.values())
        relation_hits += sum((known & candidate).values())
        conditional_hits += sum((known & oracle).values())
        relation_distribution.update(r['type'] for r in row['relations'])
    return {'entities': entity_report, 'ner_evaluated_views': len(included),
            'ner_excluded_invalid_views': len(rows) - len(included), 'attributes': attrs,
            'case': {'support': len(cases), 'accuracy': sum(a == b for a, b in cases) / len(cases) if cases else None,
                     'macro_f1_supported': sum(supported) / len(supported) if supported else None,
                     'per_class': per_class,
                     'confusion': [{'reference': a, 'prediction': b, 'count': c} for (a,b),c in Counter(cases).items()]},
            'relations': {'status': 'positive_only_partial_supervision' if relation_support else 'no_known_positive_reference',
                          'known_positive_support':relation_support,'known_positive_hits':relation_hits,
                          'known_positive_recall':relation_hits/relation_support if relation_support else None,
                          'gold_entity_known_positive_hits':conditional_hits if conditional is not None else None,
                          'gold_entity_known_positive_recall':conditional_hits/relation_support if relation_support and conditional is not None else None,
                          'reference_types':dict(relation_distribution),'precision':None,'f1': None,
                          'note':'Unlabelled pairs are unknown. Candidate edge presence is not calibrated; full P/F1 unavailable.'},
            'reference_status': sorted({r.get('reference_status','provisional') for r in rows})}


def selection_score(metrics):
    f1 = metrics['entities']['runs']['before']['overall']['micro']['f1']
    a = metrics['attributes'].values()
    support = sum(v['support'] for v in a)
    attr = sum(v['correct'] for v in metrics['attributes'].values()) / support if support else None
    values = [x for x in (f1, attr, metrics['case']['macro_f1_supported']) if x is not None]
    return sum(values) / len(values) if values else 0


def filter_entity_predictions(predictions,threshold):
    result=deepcopy(predictions)
    for row in result:
        row['entities']=[e for e in row['entities'] if e['score']>=threshold]
        ids={e['id'] for e in row['entities']}
        row['relations']=[r for r in row['relations'] if r['from_id'] in ids and r['to_id'] in ids]
    return result


def run(args, data_dir):
    out = args.output_root / 'training' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.basicConfig(filename=str(out / 'training.log'), level=logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    schema = read(data_dir / 'schema.json')
    split_names=('train','validation') if getattr(args,'validation_only',False) else ('train','validation','test')
    rows = {name: read(data_dir / (name + '.json')) for name in split_names}
    for r in rows['train']:
        r['sample_weight']=getattr(args,'synthetic_weight',1.) if r.get('synthetic') else getattr(args,'real_weight',1.)
    refine=getattr(args,'boundary_refinement',False)
    thresholds=[float(x) for x in getattr(args,'validation_thresholds','').split(',') if x] or [args.threshold]
    if any(not 0<t<1 for t in thresholds):raise ValueError('Entity thresholds must lie between 0 and 1')
    if getattr(args,'selection_metric','joint')=='ner':receipt_selection='Strict entity micro F1 on real validation only; case selected separately at fusion stage'
    else:receipt_selection='Equal mean strict entity micro F1, known-slot end-to-end attribute accuracy and supported case macro F1 on validation only'
    receipt = {'status': 'running', 'reference_status': sorted({r.get('reference_status','provisional') for split in rows.values() for r in split}), 'parameters': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'started_at': datetime.now().astimezone().isoformat(), 'device': str(device), 'torch_version': torch.__version__,
               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'working_tree_status': subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
               'inputs': {p.name: sha256(p) for p in data_dir.glob('*') if p.is_file()},
               'base_model_files': {p.name: sha256(p) for p in args.base_model.glob('*') if p.is_file()},
               'code_files': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'modernbert_ml_backend/modeling/schema_multitask.py', ROOT/'emr_annotation/training_data/schema_multitask.py', ROOT/'modernbert_ml_backend/train_schema.py']},
               'selection': receipt_selection,
               'relation_training': 'Explicit observed positive typed edges; no-edge targets only for references marked relations_complete. Unobserved real pairs remain unknown.',
               'threshold_policy': 'Fixed before any test evaluation; no test tuning'}
    write_json(out / 'experiment_receipt.json', receipt)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        datasets, audits = {}, {}
        for name, split in rows.items():
            datasets[name], audits[name] = make_windows(split, tokenizer, schema, args.max_length, args.stride, args.max_span_width,refine)
        write_json(out / 'token_audit.json', audits)
        model = SchemaMultitaskModel(args.base_model, schema, args.max_span_width,boundary_refinement=refine).to(device)
        if getattr(args,'initial_checkpoint',None):
            oldcfg=read(args.initial_checkpoint/'model_config.json')
            if oldcfg['schema']!=schema or Path(oldcfg['base_model_path']).resolve()!=args.base_model.resolve():raise ValueError('Warm start schema/base mismatch')
            previous=SchemaMultitaskModel(args.base_model,schema,oldcfg['max_span_width'],boundary_refinement=oldcfg.get('boundary_refinement',False)).to(device)
            load_schema_checkpoint(previous,args.initial_checkpoint/'best_weights.pt',device,args.base_model)
            missing,unexpected=model.load_state_dict(previous.state_dict(),strict=False)
            if unexpected or any(not (k=='width_bias' or k.startswith('boundary_heads.')) for k in missing):raise ValueError('Unexpected warm-start missing keys')
            receipt['warm_start_sha256']=sha256(args.initial_checkpoint/'best_weights.pt');del previous
        model.negative_weight=getattr(args,'negative_weight',1.)
        relation_types = sorted({r['type'] for row in rows['train'] for r in row['relations']})
        model.relation_supported_types = relation_types
        model.relation_has_negative_supervision = any(r.get('relations_complete') for r in rows['train'])
        receipt['relation_supported_types'] = relation_types
        # Keep the last-four-layer policy across base encoder architectures.
        receipt['encoder_training'] = unfreeze_encoder_tail(model.encoder,getattr(args,'encoder_layers',4))
        receipt['encoder_model_type'] = model.encoder.config.model_type
        receipt['attention_implementation'] = model.encoder.config._attn_implementation
        if tokenizer.pad_token_id != 0:
            raise ValueError('Current window collator requires pad_token_id=0')
        receipt['tokenizer_pad_token_id'] = tokenizer.pad_token_id
        compact_hashes = None
        if model.encoder.config.model_type == 'modernbert':
            compact_hashes = {name: h for name, h in receipt['base_model_files'].items()
                              if name == 'config.json' or name.endswith(('.bin', '.safetensors'))}
        receipt['checkpoint_format'] = 'frozen_encoder_delta_v1' if compact_hashes else 'full_state_dict'
        for parameter in model.relation_head.parameters():
            parameter.requires_grad = bool(relation_types)
        receipt['parameters_total'] = sum(p.numel() for p in model.parameters())
        receipt['parameters_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('RUNTIME', device, receipt['gpu'], 'windows', {k:len(v) for k,v in datasets.items()}, flush=True)
        # Warm-up is excluded from measured inference latency.
        predict(model, rows['validation'][:1], tokenizer, args, device)
        if 'test' in rows:
            before, _ = predict(model, rows['test'], tokenizer, args, device)
            before_gold, _ = predict(model, rows['test'], tokenizer, args, device, gold_attributes=True)
            before_metrics = evaluate(rows['test'], before, schema, before_gold)
            write_json(out / 'before_predictions.json', before)
            write_json(out / 'before_metrics.json', before_metrics)
        encoder_params = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
        head_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
        optimizer = torch.optim.AdamW([{'params':encoder_params,'lr':getattr(args,'encoder_lr',2e-5)}, {'params':head_params,'lr':getattr(args,'head_lr',5e-4)}], weight_decay=.01)
        loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate(b, device), generator=torch.Generator().manual_seed(args.seed))
        best_score, stale = -1, 0
        accumulation=getattr(args,'accumulation',1)
        if accumulation<1:raise ValueError('accumulation must be positive')
        total_steps=math.ceil(len(loader)/accumulation)*args.epochs
        warmup=int(total_steps*getattr(args,'warmup_fraction',0.))
        use_schedule=bool(warmup)
        def schedule(step):
            if step<warmup:return (step+1)/max(warmup,1)
            return max(.1,.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule) if use_schedule else None
        train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for step, (ids, mask, batch) in enumerate(loader):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    hidden = model.encode(ids, mask)
                    logits = model.span_logits(hidden)
                    loss = training_loss(model, hidden, logits, batch)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                group_size=min(accumulation,len(loader)-(step//accumulation)*accumulation)
                (loss/group_size).backward()
                if (step+1)%accumulation==0 or step+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
                    if scheduler:scheduler.step()
                losses.append(float(loss.detach()))
                if step % 50 == 0:
                    print('STEP',epoch,step,'loss',round(losses[-1],4),flush=True)
            saved_threshold=args.threshold;args.threshold=min(thresholds)
            validation, _ = predict(model, rows['validation'], tokenizer, args, device)
            args.threshold=saved_threshold
            candidates=[]
            for threshold in thresholds:
                filtered=filter_entity_predictions(validation,threshold)
                measures=evaluate(rows['validation'],filtered,schema)
                selected=measures['entities']['runs']['before']['overall']['micro']['f1'] if getattr(args,'selection_metric','joint')=='ner' else selection_score(measures)
                candidates.append((selected,threshold,measures,filtered))
            score,selected_threshold,metrics,selected_predictions=max(candidates,key=lambda x:x[0])
            history = {'epoch':epoch,'loss':sum(losses)/len(losses),'selection_score':score,
                       'seconds':time.perf_counter()-started,'metrics':metrics,'entity_threshold':selected_threshold,
                       'threshold_scores':[{ 'threshold':t,'selection':s} for s,t,_,_ in candidates]}
            with (out / 'validation_history.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(history,ensure_ascii=False)+'\n')
            print('EPOCH',epoch,'loss',round(history['loss'],4),'score',round(score,4),'seconds',round(history['seconds'],1),flush=True)
            if score > best_score + .001:
                best_score, stale = score, 0
                save_schema_checkpoint(model, out / 'best_weights.pt', compact_hashes)
                receipt['best_epoch'] = epoch
                receipt['best_validation_score'] = score
                receipt['selected_entity_threshold']=selected_threshold
                write_json(out/'validation_predictions.json',selected_predictions)
                write_json(out/'validation_metrics.json',metrics)
                write_json(out / 'experiment_receipt.json', receipt)
            else:
                stale += 1
            if stale >= args.patience:
                break
        receipt['training_seconds'] = time.perf_counter() - train_start
        del optimizer, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model = SchemaMultitaskModel(args.base_model, schema, args.max_span_width,boundary_refinement=refine).to(device)
        model.relation_supported_types = relation_types
        model.relation_has_negative_supervision = any(r.get('relations_complete') for r in rows['train'])
        receipt['checkpoint_reload'] = load_schema_checkpoint(model, out / 'best_weights.pt', device, args.base_model)
        write_json(out / 'model_config.json', {'schema':schema,'max_span_width':args.max_span_width,
                    'base_model_path':str(args.base_model.resolve()), 'relation_enabled':bool(relation_types),
                    'relation_supported_types':relation_types, 'relation_supervision':('real_positive_only_plus_complete_synthetic_negatives' if model.relation_has_negative_supervision else 'positive_only'),
                    'relation_has_negative_supervision': model.relation_has_negative_supervision,
                    'checkpoint_format':receipt['checkpoint_format'],
                    'threshold':receipt['selected_entity_threshold'],'max_length':args.max_length,'stride':args.stride,
                    'boundary_refinement':refine,'span_suppression_overlap':getattr(args,'span_suppression_overlap',0.)})
        tokenizer.save_pretrained(out / 'tokenizer')
        args.threshold=receipt['selected_entity_threshold']
        if 'test' not in rows:
            receipt.update(status='completed_validation_only',finished_at=datetime.now().astimezone().isoformat(),checkpoint_reloaded=True,all_checkpoint_tensors_finite=True,
                           training_seconds=receipt['training_seconds'],test_not_evaluated=True,
                           outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='experiment_receipt.json'})
            write_json(out/'experiment_receipt.json',receipt);print('COMPLETED_VALIDATION_ONLY',out,'best epoch',receipt['best_epoch'],'NER F1',best_score,flush=True);return
        predict(model, rows['validation'][:1], tokenizer, args, device)
        after, timings = predict(model, rows['test'], tokenizer, args, device)
        after_gold, _ = predict(model, rows['test'], tokenizer, args, device, gold_attributes=True)
        after_metrics = evaluate(rows['test'], after, schema, after_gold)
        write_json(out / 'after_predictions.json', after)
        write_json(out / 'after_gold_entity_predictions.json', after_gold)
        write_json(out / 'after_metrics.json', after_metrics)
        write_json(out / 'latency.json', {'rows':timings,'unique_texts':len(timings),
                    'p50_ms':float(np.percentile([t['milliseconds'] for t in timings],50)),
                    'p95_ms':float(np.percentile([t['milliseconds'] for t in timings],95)),
                    'characters_total':sum(t['characters'] for t in timings),
                    'documents_per_second':len(timings)/sum(t['milliseconds'] for t in timings)*1000,
                    'comparison_to_external_model':None})
        receipt.update(status='completed_provisional_partial_schema', finished_at=datetime.now().astimezone().isoformat(),
                       checkpoint_reloaded=True, all_checkpoint_tensors_finite=True,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                       outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='experiment_receipt.json'})
        write_json(out / 'experiment_receipt.json', receipt)
        print('COMPLETED',out,'best epoch',receipt['best_epoch'],flush=True)
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        write_json(out / 'experiment_receipt.json', receipt)
        raise
