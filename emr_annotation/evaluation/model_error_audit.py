"""Read-only, source-backed model error ledger; categories are descriptive, not causal."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess

from emr_annotation.annotation_analysis.entity_label_errors import residuals, categorize
from emr_annotation.evaluation.entity_predictions import normalize_records, score_run
from emr_annotation.evaluation.pathogen_attribute_audit import audit as pathogen_audit
from emr_annotation.evaluation.ner_re import relation_keys
from emr_annotation.training_data.schema_multitask import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'same_span_wrong_label':'同位置标签错误','same_label_wrong_boundary':'同标签范围错误',
         'overlap_label_and_boundary':'范围与标签均有差异','extra_no_overlap':'无重叠参考的误报',
         'omitted_no_overlap':'无重叠预测的漏报','duplicate_prediction':'重复预测',
         'ambiguous_same_span_labels':'同范围多标签需核查','reference_count_shortfall':'预测数量不足',
         'missing_entity':'上游实体未严格匹配','wrong_attribute':'实体匹配但属性错误',
         'wrong_case':'病例类别错分','missing_relation':'参考关系未严格匹配'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inspect(reference, predictions, schema, train=(), case_predictions=None):
    labels=set(schema['labels'])
    groups={'schema':schema['labels']}
    gold=normalize_records(reference,labels,reference=True,source='reference')
    pred=normalize_records(predictions,labels,reference=False,source='prediction')
    if set(gold)!=set(pred):raise ValueError('Missing or extra prediction tasks')
    score=score_run(gold,pred,groups)
    by_id={str(r['task_id']):r for r in predictions}
    by_case={str(r['task_id']):r for r in (case_predictions or predictions)}
    if len(by_case)!=len(case_predictions or predictions) or set(by_case)!=set(by_id):raise ValueError('Missing, duplicate or extra case prediction tasks')
    if any(c.get('text',by_id[tid]['text'])!=by_id[tid]['text'] for tid,c in by_case.items()):raise ValueError('Case prediction text mismatch')
    train_labels=Counter(e['label'] for r in train if not r.get('synthetic') for e in r['entities'])
    train_mentions=Counter((e['label'],r['text'][e['start']:e['end']]) for r in train if not r.get('synthetic') for e in r['entities'])
    ledger=[];attributes={};case_confusion=Counter();case_support=Counter();case_correct=Counter()
    case_rows=[];positive_relation_support=0;positive_relation_hits=0
    def entry(row,domain,kind,category,entity=None,**extra):
        e=entity or {};s=e.get('start',0);end=e.get('end',len(row['text']))
        ledger.append(dict(error_id=f"{row['task_id']}:{domain}:{len(ledger)+1}",task_id=row['task_id'],domain=domain,
                           direction=kind,category=category,category_name=NAMES.get(category,category),
                           label=e.get('label'),start=s,end=end,mention=row['text'][s:end],
                           context=row['text'][max(0,s-65):min(len(row['text']),end+65)],
                           evidence_status='direct_descriptive_comparison',**extra))
    for row in reference:
        tid=str(row['task_id']);p=by_id[tid]
        if p['text']!=row['text']:raise ValueError('Exact text mismatch')
        p_left,g_left,common=residuals(p['entities'],row['entities'])
        for kind,items,opposite in [('fp',p_left,row['entities']),('fn',g_left,p['entities'])]:
            for e,category,target in categorize(items,opposite,kind):
                adjacent=[dict(start=o['start'],end=o['end'],label=o['label'],mention=row['text'][o['start']:o['end']]) for o in opposite if o['start']<e['end'] and o['end']>e['start']]
                entry(row,'entity',kind,category,e,opposite=adjacent,opposite_label=target,
                      real_train_label_support=train_labels[e['label']],
                      real_train_exact_mention_support=train_mentions[(e['label'],row['text'][e['start']:e['end']])],
                      prediction_score=e.get('score'),reference_attributes=e.get('attributes',{}) if kind=='fn' else None)
        predicted={}
        for e in p['entities']:
            key=(e['start'],e['end'],e['label'])
            if key not in predicted or e.get('score',0)>predicted[key].get('score',0):predicted[key]=e
        for e in row['entities']:
            match=predicted.get((e['start'],e['end'],e['label']))
            for field,config in schema['fields'].items():
                target=e.get('attributes',{}).get(field)
                if not config['per_region'] or target is None or e['label'] not in config['scope']:continue
                answer=match.get('attributes',{}).get(field,'missing_attribute') if match else 'missing_entity'
                counts=attributes.setdefault(field,Counter())
                counts['support']+=1;counts['correct']+=answer==target
                counts['upstream_entity_error']+=match is None
                counts['matched_attribute_error']+=bool(match and answer!=target)
                if answer!=target:
                    entry(row,'attribute','error','missing_entity' if match is None else 'wrong_attribute',e,
                          field=field,reference=target,prediction=answer,prediction_score=match.get('score') if match else None)
        source_by_key={(e['start'],e['end'],e['label']) for e in p['entities']}
        actual_rel=relation_keys(p,schema['relations'],reference=False)
        known_rel=relation_keys(row,schema['relations'],reference=True)
        positive_relation_support+=sum(known_rel.values());positive_relation_hits+=sum((actual_rel&known_rel).values())
        for key,count in (known_rel-actual_rel).items():
            for _ in range(count):
                a,b,typ=key
                entry(row,'relation','fn','missing_relation',None,relation_type=typ,endpoints=[a,b],
                      endpoints_present=a in source_by_key and b in source_by_key)
        c=by_case[tid];target=row.get('case_decision');answer=c.get('case_decision')
        if target is not None:
            case_support[target]+=1;case_correct[target]+=target==answer;case_confusion[(target,answer)]+=1
            probabilities=c.get('case_probabilities',{})
            info=dict(task_id=row['task_id'],reference=target,prediction=answer,correct=target==answer,
                      reference_case_rule=row.get('case_rule'),reference_case_reason=row.get('case_reason'),
                      probabilities=probabilities,confidence=probabilities.get(answer),
                      entity_tp=sum(common.values()),entity_fp=len(p_left),entity_fn=len(g_left),
                      context=row['text'][:360],cause_status='upstream correlation is descriptive; causal attribution requires ablation')
            case_rows.append(info)
            if target!=answer:entry(row,'case','error','wrong_case',None,**{k:v for k,v in info.items() if k not in ('task_id','context')})
    counts=Counter((r['domain'],r['direction'],r['category']) for r in ledger)
    assert sum(r['domain']=='entity' and r['direction']=='fp' for r in ledger)==score['overall']['micro']['fp']
    assert sum(r['domain']=='entity' and r['direction']=='fn' for r in ledger)==score['overall']['micro']['fn']
    summary=dict(entities=score,attributes={f:dict(v) for f,v in attributes.items()},pathogen_attributes=pathogen_audit(reference,predictions,schema),
                 error_categories=[dict(domain=d,direction=k,category=c,category_name=NAMES.get(c,c),count=n) for (d,k,c),n in counts.most_common()],
                 case=dict(support=dict(case_support),correct=dict(case_correct),wrong=sum(not r['correct'] for r in case_rows),total=len(case_rows),
                           confusion=[dict(reference=a,prediction=b,count=n) for (a,b),n in case_confusion.items()]),
                 relations=dict(known_positive_support=positive_relation_support,strict_hits=positive_relation_hits,
                                recall=positive_relation_hits/positive_relation_support if positive_relation_support else None,
                                full_precision_f1_not_evaluable_without_complete_negative_reference=True),
                 metric_scope='Different domains and directions have distinct denominators; do not add FP, FN and attribute errors as independent cases.',
                 causal_claims='Categories are directly observed. Training frequency and upstream error co-occurrence are diagnostic evidence, not proven causes.')
    return summary,ledger,case_rows


def render(summary,ledger,case_rows,split):
    data=json.dumps(dict(summary=summary,ledger=ledger,cases=case_rows),ensure_ascii=False).replace('</','<\\/')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型逐项错分审查</title><style>body{font-family:Microsoft YaHei,Arial;background:#f4f7fa;color:#173449;max-width:1320px;margin:auto;padding:24px}header,article{background:white;border:1px solid #d6e0e7;border-radius:12px;padding:24px;margin-bottom:18px}h1{font-size:32px}h2{font-size:24px}p{line-height:1.7}select,input{font-size:18px;padding:12px;margin:6px;border:1px solid #9cb0bd;border-radius:7px}b{color:#116b92}.context{white-space:pre-wrap;background:#edf5f8;padding:15px;line-height:1.8}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.muted{color:#637a89}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:10px;font-size:16px}@media(max-width:700px){.pair{grid-template-columns:1fr}body{padding:12px}}</style>
<header><h1>模型逐项错分审查 · SPLIT</h1><p>参考答案保持不变。错误类别是直接比较结果；训练样本数量及上游错误仅用于提出待验证假设。当前测试为开发诊断，不能作为独立验收。</p><p id="stats"></p><select id="domain"><option value="">所有任务</option><option value="entity">实体</option><option value="attribute">实体属性</option><option value="case">病例分类</option><option value="relation">关系</option></select><select id="category"><option value="">所有错误类别</option></select><input id="search" placeholder="病例、实体、字段或原文关键词"><span id="count"></span><p><button class="prev">上一页</button> <span class="page"></span> <button class="next">下一页</button></p></header><main id="list"></main><footer><button class="prev">上一页</button> <span class="page"></span> <button class="next">下一页</button></footer><script type="application/json" id="data">DATA</script><script>
const d=JSON.parse(document.querySelector('#data').textContent),$=s=>document.querySelector(s);const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const m=d.summary.entities.overall.micro;$('#stats').textContent=`严格实体F1 ${(m.f1*100).toFixed(2)}%，误报${m.fp}，漏报${m.fn}；病例错分${d.summary.case.wrong}/${d.summary.case.total}。`;
for(const c of [...new Set(d.ledger.map(r=>r.category))]){const op=document.createElement('option');op.value=c;op.textContent=d.ledger.find(r=>r.category===c).category_name;$('#category').append(op)}
let page=0;function show(){let rows=d.ledger.filter(r=>(!$('#domain').value||r.domain===$('#domain').value)&&(!$('#category').value||r.category===$('#category').value)&&JSON.stringify(r).toLowerCase().includes($('#search').value.toLowerCase()));const pages=Math.max(1,Math.ceil(rows.length/50));page=Math.max(0,Math.min(page,pages-1));$('#count').textContent=` ${rows.length}条错误记录`;document.querySelectorAll('.page').forEach(e=>e.textContent=`第${page+1}/${pages}页`);document.querySelectorAll('.prev').forEach(e=>e.disabled=page===0);document.querySelectorAll('.next').forEach(e=>e.disabled=page===pages-1);$('#list').innerHTML=rows.slice(page*50,(page+1)*50).map(r=>`<article id="${esc(r.error_id)}"><h2>${esc(r.category_name)} · ${esc(r.label||r.field||r.relation_type||'病例')}</h2><p class="muted">${esc(r.task_id)} · ${esc(({entity:'实体',attribute:'属性',case:'病例分类',relation:'关系'})[r.domain])} · [${r.start}, ${r.end})</p><div class="context">${esc(r.context)}</div><div class="pair"><p><b>参考</b><br>${esc(r.reference??(r.direction==='fn'?r.mention:JSON.stringify(r.opposite)))}</p><p><b>预测</b><br>${esc(r.prediction??(r.direction==='fp'?r.mention:JSON.stringify(r.opposite)))}</p></div><p>${r.field?'属性字段：'+esc(r.field):''} ${r.real_train_label_support!==undefined?'真实训练标签数量：'+r.real_train_label_support+'；同词数量：'+r.real_train_exact_mention_support:''}</p><details><summary>完整证据和置信度</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></article>`).join('')}
for(const s of ['#domain','#category','#search'])$(s).addEventListener('input',()=>{page=0;show()});document.querySelectorAll('.prev').forEach(e=>e.onclick=()=>{page--;show();window.scrollTo(0,0)});document.querySelectorAll('.next').forEach(e=>e.onclick=()=>{page++;show();window.scrollTo(0,0)});show();</script></html>'''.replace('SPLIT',html.escape(split)).replace('DATA',data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data-dir','extractor-predictions','case-predictions','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    sources=[a.data_dir/(a.split+'.json'),a.data_dir/'train.json',a.data_dir/'schema.json',a.extractor_predictions,a.case_predictions]
    summary,ledger,cases=inspect(read(sources[0]),read(a.extractor_predictions),read(sources[2]),read(sources[1]),read(a.case_predictions))
    write_json(out/'summary.json',summary);write_json(out/'errors.json',ledger);write_json(out/'case_comparison.json',cases)
    (out/'review.html').write_text(render(summary,ledger,cases,a.split),encoding='utf-8')
    write_json(out/'audit_manifest.json',dict(status='completed',stage='evaluation',split=a.split,started_after='2026-10-12T00:50:00+08:00',created_at=datetime.now(timezone.utc).isoformat(),
               source_sha256={str(s.resolve()):sha256(s) for s in sources},code_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))},
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),error_count=len(ledger),
               outputs={s.name:sha256(s) for s in out.iterdir() if s.is_file()}))
    print(json.dumps(dict(entity=summary['entities']['overall']['micro'],categories=summary['error_categories'],attributes=summary['attributes'],case=summary['case'],relations=summary['relations']),ensure_ascii=False))


if __name__=='__main__':main()
