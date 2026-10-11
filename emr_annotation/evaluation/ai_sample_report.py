"""Six-section report for a locally adjudicated AI sample experiment."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

from emr_annotation.evaluation.entity_predictions import normalize_records, build_report
from emr_annotation.evaluation.regex_entity_baseline import compile_rules, run_baseline
from emr_annotation.evaluation.schema_training_report import FIELD_NAMES, percent
from emr_annotation.training_data.schema_multitask import write_json, sha256


def build(data_dir, training_dir, adjudication_dir, delivery_dir, rules_path, app_dir):
    read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
    schema=read(data_dir/'schema.json'); audit=read(data_dir/'preparation_audit.json')
    split=read(data_dir/'split_manifest.json'); after=read(training_dir/'after_metrics.json')
    receipt=read(training_dir/'experiment_receipt.json'); latency=read(training_dir/'latency.json')
    points=read(adjudication_dir/'point_summary.json'); test=read(data_dir/'test.json')
    tokens=read(training_dir/'token_audit.json'); allrows=read(data_dir/'reference.json')
    delivery_dir.mkdir(parents=True,exist_ok=False)
    evaluation=delivery_dir.parents[1]/'evaluation'/delivery_dir.name
    evaluation.mkdir(parents=True,exist_ok=True)
    regex,rt=run_baseline(test,schema['controls'],compile_rules(read(rules_path),schema['controls']),3,1)
    write_json(evaluation/'regex_predictions.json',regex);write_json(evaluation/'regex_latency.json',rt)
    refs=[r for r in test if r['ner_supervised']]; ids={r['task_id'] for r in refs}
    labels=set(schema['labels'])
    norm=lambda rows:normalize_records([r for r in rows if r['task_id'] in ids],labels,reference=False,source='AI_sample_prediction')
    comparison=build_report(normalize_records(refs,labels,reference=True,source='AI_sample_reference'),
                            norm(read(training_dir/'before_predictions.json')),schema['controls'],
                            after=norm(read(training_dir/'after_predictions.json')),regex=norm(regex))
    comparison['notes']=['AI case reasoning plus rule-assisted point decisions; no expert gold.',
                         'Strict entity evaluation only in cases without pending source points.',
                         'This batch is a previously inspected development sample, not a medical blind test.']
    write_json(evaluation/'entity_comparison.json',comparison)
    metrics={k:v['overall']['micro'] for k,v in comparison['runs'].items()}
    support=sum(v['support'] for v in after['attributes'].values())
    ac=sum(v['correct'] for v in after['attributes'].values())
    oc=sum(v['gold_entity_correct'] for v in after['attributes'].values())
    target=after['case']['per_class']['目标病例信号'];rel=after['relations']
    history=[json.loads(line) for line in (training_dir/'validation_history.jsonl').read_text('utf-8').splitlines()]
    training_labels={e['label'] for r in read(data_dir/'train.json') for e in r['entities']}
    previous=read(evaluation/'previous_model_on_ai_metrics.json') if (evaluation/'previous_model_on_ai_metrics.json').exists() else None
    prevf=previous['entities']['runs']['before']['overall']['micro']['f1'] if previous else None
    supported=comparison['runs']['after']['overall']['supported_label_count']
    summary=(f'已完成182例AI裁决样例、逐点追溯、患者划分冻结及真实GPU训练。相对样例参考，'
             f'{len(refs)}例可评价测试病例的严格实体F1为{percent(metrics["after"]["f1"])}；'
             f'27例病例四分类准确率{percent(after["case"]["accuracy"])}，目标病例正确识别{target["tp"]}/{target["support"]}。\n\n'
             f'7,006条原始结果均登记处理，共保留{points["pending_points"]}个待复核点，{points["ner_masked_cases"]}例屏蔽实体损失与实体评分。'
             '本轮完成流程样例验证；病例结论经AI复核，实体、属性和关系由字面规则辅助处理，专家签署为0，指标反映对样例答案的拟合程度。')
    sections=[
      ('construction','一、建设思路回顾',
       '沿用电子病历结构化标注、双人核查、可追溯裁决、训练和评估的建设流程。本轮将现有双人标注转为逐病例、逐原始结果可核查的AI样例参考，再验证模型训练、保存、重载和全文预测。输出供人工复核和公共卫生病例信号研究。'),
      ('guide','二、数据集标注指南研发过程',
       '裁决依据为当前维护指南与字典，字段合同为pneumonia_config.xml快照：49类实体、9个区域属性、5种关系及病例四分类。未修改指南或历史导出。\n\n'
       '病例理由逐例登记索引事件、直接肺部证据和B类复核条件。点级规则检查原文最小跨度、指标与数值分离、否定作用域、病原名与诊断区分、同组检验属性以及关系端点。双方一致也核对定义，不按多数票裁决。\n\n'
       '已纠正数值/时间被标为发热或肺炎、上呼吸道感染被标为肺炎、产酸克雷伯菌被标为肺炎克雷伯菌等问题；旧记录缺关系类型时不直接补类，另建有文字依据的样例关系。当前XML包含抗体IgG而指南列表不完全同步，记录差异并以本轮XML约束有效值。\n\n'
       f'生成182×49={182*49}条标签覆盖记录，属于全文字面扫描与原点对账；无命中不能证明专家已核查不存在。{points["pending_points"]}个含糊或未解决点保留待复核，未签成医学金标。'),
      ('tools','三、数据集工具与模型建设',
       '新增本地AI样例裁决入口、逐点工作台、Label Studio predictions导入文件和保持原患者边界的数据冻结入口。原始结果与最终样例实体、属性、关系均有点号、处理和依据；病例原文只出现在本地受控数据与工作台。\n\n'
       '模型沿用中文BERT共享编码器、49类多标签跨度实体头、9个按作用域启用的属性头、全文窗口病例四分类及实体对关系头。冻结前8层、微调后4层与任务头，窗口384 token、重叠96 token、实体宽度32 token、batch 4、seed 42、阈值0.7。编码器/任务头学习率分别2e-5/5e-4。\n\n'
       f'使用{receipt["gpu"]}实际训练{len(history)}轮，验证集选第{receipt["best_epoch"]}轮；联合选模使用实体F1、已知属性端到端正确比例及病例macro F1，不按本轮测试选轮次。最佳权重已严格重载，checkpoint全部张量有限。关系仅学习已知正例类型，未标对不作负例，边是否成立尚未校准。'),
      ('annotation','四、标注情况与计划',
       f'12份原始导出合并为182例、364份原始标注，涉及106个患者组。AI样例每例只保留一份：目标病例23例、待专业复核21例、非目标79例、信息不足59例。共{points["entities"]}个样例实体及{points["relations"]}条关系正例。\n\n'
       f'7,006条原始结果均有处理记录，加上最终属性及新增点共{points["point_rows"]}个追溯点。处理计数为：'+ '、'.join(f'{k}{v}' for k,v in points['decisions'].items())+'。删除随属表示原实体重划后不沿用旧属性ID，新实体属性另立点，不等于该属性不存在。\n\n'
       '训练/验证/测试沿用上一轮相同患者和全文边界：128/27/27例、73/16/17个患者组。全部182例参与病例任务；未决原始点或来源归属冲突的33例屏蔽实体损失和实体评价，仍可保留合法已知属性及关系监督。\n\n'
       f'训练样例含{len(training_labels)}/49类，实体测试有效参考可评价{supported}/49类。未出现类别、选填空白和未标关系不计正确。所有病例由AI给出样例理由，点级使用规则辅助和例外修订，并非专家逐点签署。'),
      ('results','五、初步结果（质量和速度）',
       f'严格实体评价采用{len(refs)}例、{metrics["after"]["support"]}个样例实体：训练后精确率{percent(metrics["after"]["precision"])}、召回率{percent(metrics["after"]["recall"])}、F1 {percent(metrics["after"]["f1"])}，TP/FP/FN为{metrics["after"]["tp"]}/{metrics["after"]["fp"]}/{metrics["after"]["fn"]}。固定正则F1为{percent(metrics["regex"]["f1"])}；训练前随机任务头F1为{percent(metrics["before"]["f1"])}。'
       +(f'上一轮权重的冻结预测在同一AI参考上重评分，实体F1为{percent(prevf)}；该比较同时改变监督数据和训练轮次，不能作单因素因果结论。' if previous else '')+'\n\n'
       f'病例四分类对27例评分：准确率{percent(after["case"]["accuracy"])}，macro F1 {percent(after["case"]["macro_f1_supported"])}。目标病例正确识别{target["tp"]}/{target["support"]}，召回率{percent(target["recall"])}；逐类支持数与分数见附表。不能把原始双标参考下旧分数与AI参考下新分数直接视为提升。\n\n'
       f'属性测试有{support}个已知样例槽位，参考实体条件下正确{oc}/{support}（{percent(oc/support if support else None)}），预测实体条件下正确{ac}/{support}（{percent(ac/support if support else None)}）。参考实体条件不是临床专家金标；端到端要求跨度、类型与属性值同时正确，不是完整属性F1。\n\n'
       f'关系测试已知正例{rel["known_positive_support"]}条，端到端命中{rel["known_positive_hits"]}条，已知正例召回{percent(rel["known_positive_recall"])}；参考实体条件下召回{percent(rel["gold_entity_known_positive_recall"])}。未标实体对仍未知，不能计算完整关系精确率/F1，也不能将所有合法端点候选认定为成立。\n\n'
       f'重载预热后27篇全文本机耗时P50 {latency["p50_ms"]:.2f}毫秒、P95 {latency["p95_ms"]:.2f}毫秒，串行约{latency["documents_per_second"]:.2f}篇/秒。包括分词、全部窗口GPU编码及解码，不包含服务网络、排队和读写；只测一次。测试token审计有{tokens["test"]["unrepresentable_entities"]}/{tokens["test"]["entities"]}个样例实体不可由当前完整token边界/宽度表示。\n\n'
       '结果用于验证样例流程及对AI标签的拟合。裁决规则含字面匹配偏好，可能使同类模型/正则偏高；原批次已用于开发和多次观察，不能称独立盲验。医学正确性、近重复审计、跨机构泛化、真实Label Studio界面回写、服务并发与外部大模型对照均未验收。'),
      ('next','六、下一步计划',
       '先由专家核查病例边界、同次住院旧证据以及剩余未决点，再签署实体、属性与关系完整性；重点抽查规则补出的检验日期、否定传播、影像来源及低频实体。对当前标签覆盖记录补做专家全文核查，而不是把未命中当成已确认阴性。\n\n'
       '形成专家最终参考后另建版本，补充关系完整正负和缺少的起始/结束时间类型。按患者与近重复边界收集未参与本轮开发的独立测试集，再评估质量和多次速度、服务并发及指定通用大模型对照。当前模型可在本地生成待复核候选，正式预标注上线仍需相应验收。')]
    queries={}
    def q(name,rows,files,definition,components):
        queries[name]={'rows':rows,'source':{'type':'local_files','files':files,
                    'evidenceFlow':[{'title':'本地AI样例试验留证','detail':definition}],
                    'metricDefinitions':[{'label':name,'definition':definition,'componentIds':components}]}}
    q('preparation',[points],['point_summary.json','preparation_audit.json','split_manifest.json'],audit['policy'],['report-summary','construction','annotation'])
    q('schema',[{'entities':49,'attribute_fields':9,'relations':5,'case_classes':4}],['label_config.xml'],'Exact XML snapshot, not expert validation.',['guide'])
    q('model',[{'best_epoch':receipt['best_epoch'],'parameters':receipt['parameters_total'],'trainable_parameters':receipt['parameters_trainable']}],['experiment_receipt.json'],'Patient splits preserved; validation-only checkpoint selection.',['tools'])
    score_rows=[{'method':{'regex':'固定正则','before':'随机任务头','after':'AI样例训练后'}[k],**v,'F1Rate':v['f1']} for k,v in metrics.items()]
    if previous:
        v=previous['entities']['runs']['before']['overall']['micro'];score_rows.insert(2,{'method':'旧权重同参考重评',**v,'F1Rate':v['f1']})
    q('scores',score_rows,['entity_comparison.json','previous_model_on_ai_metrics.json'],f'Strict exact spans+labels against {len(refs)} included AI sample cases; {metrics["after"]["support"]} reference entities. No clinical gold.',['results','next','entity-overall'])
    perlabels=[{'label':l,'support':comparison['runs']['after']['per_label'][l]['support'],**{k+'_f1':comparison['runs'][k]['per_label'][l]['f1'] for k in ('regex','before','after')}} for l in schema['labels']]
    q('labels',perlabels,['entity_comparison.json'],'All 49 labels, unsupported labels null.',['label-table'])
    q('attributes',[{'field':FIELD_NAMES[f],'support':v['support'],'gold_accuracy':v['gold_entity_known_slot_accuracy'],'end_to_end_accuracy':v['end_to_end_known_slot_accuracy']} for f,v in after['attributes'].items()],['after_metrics.json'],'Known AI slots only; reference entity conditional and exact entity end-to-end; not full attribute F1.',['attribute-table'])
    q('case_classes',[{'class':k,'support':v['support'],'precision':v['precision'],'recall':v['recall'],'F1Rate':v['f1']} for k,v in after['case']['per_class'].items()],['after_metrics.json'],'27 AI-reviewed sample case decisions, not annotator view duplicates.',['case-table'])
    relation_rows=[{'type':name,**{sn:sum(r['type']==name for row in read(data_dir/(sn+'.json')) for r in row['relations']) for sn in ('train','validation','test')},'test_F1':None} for name in schema['relations']]
    q('relation_support',relation_rows,['train.json','validation.json','test.json','after_metrics.json'],'Only known positive edges; edge existence uncalibrated, full precision/F1 unavailable.',['relation-table'])
    q('latency',[{k:r[k] for k in ('characters','windows','milliseconds')} for r in latency['rows']],['latency.json'],'27 full texts; one local serial GPU pass including tokenization+encoding+decode.',['latency-chart'])
    q('training',[{k:r[k] for k in ('epoch','loss','selection_score','seconds')} for r in history],['validation_history.jsonl'],'Loss and validation-only selection; not a test learning curve.',['training-chart'])
    snapshot={'id':read(app_dir/'src/data.json')['id'],'surface':'report','title':'AI裁决样例已跑通肺炎结构化模型流程',
              'status':'observed_ai_sample_not_expert_gold','generatedAt':datetime.now(timezone.utc).isoformat(),'buildStatus':'complete','filters':[],
              'report':{'summary':summary,'sections':[{'id':i,'title':t,'text':'## '+t+'\n\n'+body} for i,t,body in sections]},'queries':queries}
    write_json(delivery_dir/'reviewed.json',snapshot);write_json(app_dir/'src/data.json',snapshot)
    lines=['# '+snapshot['title'],'','## 本轮结论','',summary,'']
    lines.extend('## '+title+'\n\n'+body+'\n' for _,title,body in sections)
    lines.extend(['## 逐实体附表','','| 标签 | 参考数 | 固定正则F1 | 随机任务头F1 | AI样例训练后F1 |','|---|---:|---:|---:|---:|'])
    lines.extend(f'| {r["label"]} | {r["support"]} | {percent(r["regex_f1"])} | {percent(r["before_f1"])} | {percent(r["after_f1"])} |' for r in perlabels)
    lines.extend(['','## 属性附表','','| 属性 | 已知槽位 | 参考实体条件 | 端到端 |','|---|---:|---:|---:|'])
    lines.extend(f'| {r["field"]} | {r["support"]} | {percent(r["gold_accuracy"])} | {percent(r["end_to_end_accuracy"])} |' for r in queries['attributes']['rows'])
    lines.extend(['','## 病例分类附表','','| 类别 | 参考数 | 精确率 | 召回率 | F1 |','|---|---:|---:|---:|---:|'])
    lines.extend(f'| {r["class"]} | {r["support"]} | {percent(r["precision"])} | {percent(r["recall"])} | {percent(r["F1Rate"])} |' for r in queries['case_classes']['rows'])
    lines.extend(['','## 可核查产物','','- [逐点裁决工作台](../../adjudication/run_001/proposal_pass_004/review.html)',
                 '- [逐点处理JSON](../../adjudication/run_001/proposal_pass_004/point_decisions.json)',
                 '- [冻结划分](../../training_data/run_003/split_manifest.json)',
                 '- [训练回执](../../training/run_003/experiment_receipt.json)',
                 '- [预测与评分](../../training/run_003/after_metrics.json)',
                 '- [运行校验](../../evaluation/run_003/runtime_verification.json)'])
    (delivery_dir/'report.md').write_text('\n'.join(lines)+'\n','utf-8')
    sources=[data_dir/'preparation_audit.json',data_dir/'split_manifest.json',training_dir/'experiment_receipt.json',training_dir/'after_metrics.json',adjudication_dir/'point_summary.json',rules_path]
    write_json(delivery_dir/'report_sources.json',{'sources':{str(p.resolve()):sha256(p) for p in sources},'generator_sha256':sha256(__file__),'medical_gold':False,'privacy_gate_deferred_by_user':True})
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','training-dir','adjudication-dir','delivery-dir','rules','app-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.data_dir,a.training_dir,a.adjudication_dir,a.delivery_dir,a.rules,a.app_dir)


if __name__=='__main__':
    main()
