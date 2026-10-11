"""Build a six-part evidence report from a provisional schema experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from emr_annotation.evaluation.entity_predictions import normalize_records, build_report
from emr_annotation.evaluation.regex_entity_baseline import compile_rules, run_baseline
from emr_annotation.training_data.schema_multitask import sha256, write_json


FIELD_NAMES = {'finding_context':'存在状态','temporality':'时间属性','clinical_course':'病程变化',
               'conditional':'条件性','hypothetical':'假设性','experiencer':'主体','test_result':'检测结果',
               'specimen_type':'标本类型','detection_method':'检测方式'}


def percent(value):
    return '不可评价' if value is None else f'{value:.2%}'


def build(data_dir, training_dir, delivery_dir, rules_path, app_dir=None):
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
    schema = read(data_dir/'schema.json')
    audit = read(data_dir/'preparation_audit.json')
    split = read(data_dir/'split_manifest.json')
    receipt = read(training_dir/'experiment_receipt.json')
    tokens = read(training_dir/'token_audit.json')
    test = read(data_dir/'test.json')
    before = read(training_dir/'before_metrics.json')
    after = read(training_dir/'after_metrics.json')
    latency = read(training_dir/'latency.json')
    rules = compile_rules(read(rules_path), schema['controls'])
    regex, regex_time = run_baseline(test, schema['controls'], rules, 3, 1)
    eval_dir = delivery_dir.parent.parent/'evaluation'/delivery_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir/'regex_predictions.json', regex)
    write_json(eval_dir/'regex_latency.json', regex_time)
    refs = [r for r in test if r['ner_supervised']]
    ids = {r['task_id'] for r in refs}
    labels = set(schema['labels'])
    normalize = lambda rows: normalize_records([r for r in rows if r['task_id'] in ids], labels, reference=False, source='trial_prediction')
    comparison = build_report(normalize_records(refs, labels, reference=True, source='unadjudicated_views'),
                              normalize(read(training_dir/'before_predictions.json')), schema['controls'],
                              after=normalize(read(training_dir/'after_predictions.json')), regex=normalize(regex))
    comparison['notes'] = ['Provisional double-annotation views, not clinical gold.',
                           'Strict spans and labels; invalid-span views excluded. No patient or identical-text overlap.',
                           'No near-duplicate audit. Test labels not used to tune thresholds, epochs or regex rules.']
    write_json(eval_dir/'entity_comparison.json', comparison)
    overall = {k: v['overall']['micro'] for k,v in comparison['runs'].items()}
    train_rows = read(data_dir/'train.json')
    train_labels = {e['label'] for r in train_rows for e in r['entities']}
    support = sum(a['support'] for a in after['attributes'].values())
    attr_correct = sum(a['correct'] for a in after['attributes'].values())
    attr_oracle = sum(a['gold_entity_correct'] for a in after['attributes'].values())
    total_relations = audit['exclusions'].get('invalid_relations',0)
    retained_relations = sum(v for k,v in audit['support'].items() if k.startswith('relation:'))
    relation_types = receipt.get('relation_supported_types', [])
    relation_description = ('关系分支训练仅使用原始已标类型关系的唯一Schema兼容端点投影，'
                            '其余实体对不参与关系损失。推理输出合法端点的类型候选；未学习边存在概率，不能自动认定这些候选均成立。') if relation_types else '关系头已定义，但本轮无有效类型监督，停用训练及预测。'
    input_rows = sum(source['task_rows'] for source in audit['sources'])
    supported_labels = comparison['runs']['after']['overall']['supported_label_count']
    counts = [split['splits'][name] for name in ('train','validation','test')]
    split_texts = '/'.join(str(c['unique_texts']) for c in counts)
    split_patients = '/'.join(str(c['patients']) for c in counts)
    split_views = '/'.join(str(c['tasks']) for c in counts)
    target_class = after['case']['per_class']['目标病例信号']
    sections = [
      ('construction','一、建设思路回顾',
       '本项目围绕中文电子病历建立规范化标注、可追溯数据准备、结构化抽取和人工复核流程，为公共卫生病例信号分析提供依据。本轮使用已到位的真实导出开展小规模模型试验，验证从Schema到数据、训练、选模和评估的链路。输出定位为待复核候选，不能直接承担医学诊断。'),
      ('guide','二、数据集标注指南研发过程',
       '当前维护源为documentation/annotation/下的指南和字典，模型合同来自本轮保存的pneumonia_config.xml快照。XML定义49个实体、9个实体属性字段、5种关系和病例四分类。指南版本、原始双标和模型结果分别留证；本轮没有开展新的专家指南验收。\n\n'
       f'导出审计发现{audit["issue_counts"].get("span_text_mismatch",0)}处选区文字不一致、{audit["issue_counts"].get("invalid_offsets",0)}处非法偏移、{audit["issue_counts"].get("multi_label_region",0)}项同区域多标签提示。这些是结构审计计数，不等于独立医学错误数。重叠及同跨度标签被模型保留，不自动作医学取舍；标签与用语是否正确仍需专家复核。'),
      ('tools','三、数据集工具与模型建设',
       '复用已有Schema解析、双标导出整合、患者与全文连通分组及严格实体评分。新增独立中文多任务试验入口，未替换既有ModernBERT服务权重。\n\n'
       '**模型结构：**中文BERT共享编码器→有限宽度多标签跨度头；实体起止、跨度均值与类型表示→9个按作用域启用的属性头；窗口CLS表示→病例四分类，全文窗口logits平均；两实体表示与中间语境→关系分类头。'+relation_description+'病例头第一版使用全文窗口信息，尚未实现结构化证据融合。\n\n'
       '基础权重使用[google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，下载绑定精确revision和文件哈希。冻结编码器前8层，微调后4层及任务头。384-token窗口、96-token重叠、最大32-token实体跨度；支持相互重叠的多标签实体，不把全文截为前384个token。\n\n'
       f'训练使用{receipt["gpu"]}，batch size 4，编码器学习率2e-5、任务头5e-4，AdamW；固定seed 42和实体阈值0.7。最多12轮，选中第{receipt["best_epoch"]}轮；验证选模指标为严格实体micro F1、已知属性槽位端到端正确率和病例macro F1的等权均值。未用测试结果调参。'),
      ('annotation','四、标注情况与计划',
       f'{len(audit["sources"])}份文件共{input_rows}条快照任务记录，重复提交内容去重后{audit["unique_tasks"]}例、{audit["annotation_views"]}份标注，涉及{split["patients"]}名患者。本轮保留两名标注者的答案，均未转为最终裁决金标；标注视图数不能当作独立病例数。\n\n'
       f'训练/验证/测试为{split_texts}例原文，{split_patients}名患者，对应{split_views}份标注视图。同一患者及完全相同全文的记录不跨集合；近重复尚未核查。训练部分覆盖{len(train_labels)}/49类实体，测试可评价{supported_labels}/49类，其余无测试参考的类别以空值显示，不计100%。\n\n'
       f'自动转换排除{audit["exclusions"].get("invalid_entities",0)}个无效实体记录及{audit["exclusions"].get("duplicate_entities",0)}个重复跨度标签，屏蔽{audit["exclusions"].get("invalid_attribute_slots",0)}个无效属性槽位。无效实体所在视图的实体损失被屏蔽，保留有效的其他任务监督；未填属性不视为阴性。关系保留{retained_relations}条明确类型且可唯一解析端点的原始正例，屏蔽{total_relations}条无法解析记录。不能猜测补类或将未标关系当作无关系。\n\n'
       '后续标注先回读偏移和同区域多标签，再逐项补齐关系类型、方向和端点，保留裁决人、依据和日期。最终裁决完成后另建输入版本和运行编号，保留本轮未裁决结果。'),
      ('results','五、初步结果',
       f'**抽取质量仍需改善。**实体评分使用{len(refs)}份有效测试标注视图，共{overall["after"]["support"]}个参考实体：训练后严格精确率{percent(overall["after"]["precision"])}、召回率{percent(overall["after"]["recall"])}、F1 {percent(overall["after"]["f1"])}；正则F1 {percent(overall["regex"]["f1"])}，差值{(overall["after"]["f1"]-overall["regex"]["f1"])*100:.2f}个百分点。训练前是预训练编码器加随机任务头，F1为0，不能理解为旧版生产模型表现。训练后有{overall["after"]["tp"]}个正确抽取、{overall["after"]["fp"]}个误报和{overall["after"]["fn"]}个漏报，不适合据此宣称准确率达到业务要求。\n\n'
       f'病例四分类对{after["case"]["support"]}份标注视图评分，准确率{percent(after["case"]["accuracy"])}、macro F1 {percent(after["case"]["macro_f1_supported"])}。其中“目标病例信号”正确识别{target_class["tp"]}/{target_class["support"]}，召回率{percent(target_class["recall"])}；总体准确率不能掩盖这一类别的漏检。“待专业复核”逐类支持与F1见附表。所有指标均相对未裁决的个人答案，不能等同于临床准确率。\n\n'
       f'属性共{support}个有合法已知值的测试槽位：人工实体条件下正确{attr_oracle}/{support}（{percent(attr_oracle/support)}），预测实体条件下正确{attr_correct}/{support}（{percent(attr_correct/support)}）。端到端要求实体跨度、实体类别和属性值同时正确；不以未标属性作为负例，不把该正确槽位比例称为完整属性F1。部分字段只有单一或高度偏斜取值，高正确率不能证明少数属性已学会。\n\n'
       f'模型使用所有窗口覆盖全文，但测试有{tokens["test"]["unrepresentable_entities"]}/{tokens["test"]["entities"]}个保留实体不能精确表示为32-token以内的完整token边界跨度，这些实体仍进入适用的评分分母。\n\n'
       f'**速度为本机单次试验。**重载模型并预热后，27篇独立测试全文的端到端耗时P50为{latency["p50_ms"]:.2f}毫秒，P95为{latency["p95_ms"]:.2f}毫秒；串行处理约{latency["documents_per_second"]:.2f}篇/秒。包括分词、全部窗口GPU编码和解码，未包含服务网络、排队或文件读写。仅一次测量，尚未测服务并发、长期稳定性及指定通用大模型对照，不能据此推算全院吞吐量。\n\n'
       f'关系训练支持类型为{relation_types}；当前患者分组下，保留的关系正例全部属于训练集合，验证和测试没有合格关系正例。关系头参数训练不等于关系效果已验证，不能报告关系P/R/F1；起始时间、结束时间没有训练正例。服务路由、Label Studio回写、页面兼容和医学效果没有在本轮验收。'),
      ('next','六、下一步计划',
       '优先完成最终医学裁决和关系补标，确认每个项目实际采用的XML。随后按已有测试患者边界审计近重复；若形成全新的评估协议，另行登记冻结版本，避免重复使用本轮测试结果调参。\n\n'
       '基于裁决训练集与验证集分析误报来源、过长跨度和低频标签，确定实体宽度与阈值；补充低频实体、少数属性及“待专业复核”样本。关系需补充完整正负参考、验证/测试支持，以及起始时间和结束时间样本，再开展边存在判断和端到端评分；病例分支再检验全文与结构化证据融合是否有效。\n\n'
       '在固定测试集和硬件条件下开展多次速度测量、服务并发试验及选定通用大模型对照。通过质量和服务验证后再用于人工预标注，保留人工确认。')]
    sections = [(identity, title, body.replace('**', '')) for identity, title, body in sections]
    delta = (overall['after']['f1']-overall['regex']['f1'])*100
    summary = (f'本轮完成182例双人原始标注的中文跨度模型试验训练及权重重载。测试实体F1为{percent(overall["after"]["f1"])}，较现有正则高{delta:.2f}个百分点；病例macro F1为{percent(after["case"]["macro_f1_supported"])}。\n\n'
               f'当前实体误报较多，参考尚未裁决。关系分支利用{retained_relations}条原始已标类型正例进行了试验学习，但缺少完整正负及独立测试参考。本报告可作为模型建设和初步试验进展材料，完整医学效果验收仍需最终参考和关系补标。')
    previous_dir=training_dir.parent/'run_001'
    if training_dir.name!='run_001' and (previous_dir/'after_metrics.json').exists():
        old=read(previous_dir/'after_metrics.json')
        old_f1=old['entities']['runs']['before']['overall']['micro']['f1']
        summary+=f'\n\n首轮仅实体/属性/病例试验的实体F1为{percent(old_f1)}；本轮加入已标关系正例后为{percent(overall["after"]["f1"])}，变化{(overall["after"]["f1"]-old_f1)*100:+.2f}个百分点。两次均由各自验证集选轮次，不能将这一观察归因于关系训练的因果效果，也没有按测试分数选择生产模型。原始划分保持一致；本批测试已用于多次开发试验，不再视作新的盲验。'
    queries = {}
    def query(name, rows, files, definition, components):
        queries[name] = {'rows':rows,'source':{'type':'local_files','files':files,
                         'evidenceFlow':[{'title':'可核查的本地试验产物','detail':definition}],
                         'metricDefinitions':[{'label':name,'definition':definition,'componentIds':components}]}}
    query('preparation',[{'cases':audit['unique_tasks'],'views':audit['annotation_views'],'patients':split['patients'],
                          'invalid_relations':total_relations}],['preparation_audit.json','split_manifest.json'],audit['policy'],['annotation','report-summary'])
    query('model',[{'best_epoch':receipt['best_epoch'],'parameters':receipt['parameters_total'],
                    'trainable_parameters':receipt['parameters_trainable']}],['experiment_receipt.json','model_config.json'],
          'Validation-only model selection; full checkpoint reloaded before held-out prediction.', ['tools'])
    query('schema',[{'entities':len(schema['labels']),'attribute_fields':len(FIELD_NAMES),
                    'relations':len(schema['relations']),'case_classes':len(schema['fields']['case_decision']['options'])}],
          ['label_config.xml'],'Exact XML snapshot defining labels, choice scopes and relation types.', ['guide'])
    query('scores',[{'method':{'regex':'正则','before':'训练前初始化','after':'训练后'}[k],
                    'precision':v['precision'],'recall':v['recall'],'F1Rate':v['f1'],'support':v['support'],
                    'TP':v['tp'],'FP':v['fp'],'FN':v['fn']} for k,v in overall.items()],
          ['entity_comparison.json'],'Strict exact character spans and entity labels, 49 valid annotation views with 294 entities. Unadjudicated, not independent 49 cases.', ['results','entity-overall'])
    per_labels = [{'label':l,'support':comparison['runs']['after']['per_label'][l]['support'],
                  **{k+'_f1':comparison['runs'][k]['per_label'][l]['f1'] for k in ('regex','before','after')}} for l in schema['labels']]
    query('labels',per_labels,['entity_comparison.json'],'49 configured entity labels; F1 null without reference support.', ['label-table'])
    query('attributes',[{'field':FIELD_NAMES[f],'support':v['support'],'gold_accuracy':v['gold_entity_known_slot_accuracy'],
                         'end_to_end_accuracy':v['end_to_end_known_slot_accuracy']} for f,v in after['attributes'].items()],
          ['after_metrics.json'],'Known valid slots only; end-to-end requires exact entity span and label. Not complete attribute F1.', ['attribute-table'])
    query('case_classes',[{'class':c,'support':v['support'],'precision':v['precision'],'recall':v['recall'],'F1Rate':v['f1']} for c,v in after['case']['per_class'].items()],
          ['after_metrics.json'],'Four-class scores against 54 unadjudicated annotator views of 27 cases.', ['case-table'])
    query('latency',[{k:r[k] for k in ('characters','windows','milliseconds')} for r in latency['rows']],
          ['latency.json'],'27 unique full texts, one serial GPU pass after warmup, includes tokenization, encoder windows and decode; excludes service network.', ['latency-chart'])
    history=[json.loads(line) for line in (training_dir/'validation_history.jsonl').read_text('utf-8').splitlines()]
    query('training',[{k:r[k] for k in ('epoch','loss','selection_score','seconds')} for r in history],
          ['validation_history.jsonl'],'Training loss and validation-only selection history. Test never used for selection.', ['training-chart'])
    relation_rows=[]
    for name in schema['relations']:
        counts_by_split={split_name:sum(r['type']==name for row in read(data_dir/(split_name+'.json')) for r in row['relations']) for split_name in ('train','validation','test')}
        relation_rows.append({'type':name,**counts_by_split,'test_F1':None})
    query('relation_support',relation_rows,['train.json','validation.json','test.json'],
          'Only explicit typed edges with unique compatible projections retained; unknown pairs never negatives. F1 null without complete held-out reference.', ['relation-table'])
    snapshot={'surface':'report','title':'肺炎病历结构化模型试验与建设进展','status':'observed_provisional',
              'generatedAt':datetime.now(timezone.utc).isoformat(),'buildStatus':'complete','filters':[],
              'report':{'summary':summary,'sections':[{'id':i,'title':t,'text':f'## {t}\n\n{body}'} for i,t,body in sections]},'queries':queries}
    app_data=(app_dir or delivery_dir/'report_app')/'src/data.json'
    if app_data.exists():
        snapshot['id']=read(app_data)['id']
    write_json(delivery_dir/'reviewed.json',snapshot)
    if app_data.exists():
        write_json(app_data,snapshot)
    text=['# 肺炎病历结构化模型试验与建设进展','','## 本轮结论','',summary,'']
    text.extend(f'## {title}\n\n{body}\n' for _,title,body in sections)
    text.extend(['## 逐标签附表','','| 标签 | 测试参考数 | 正则F1 | 训练前F1 | 训练后F1 |','|---|---:|---:|---:|---:|'])
    text.extend(f'| {r["label"]} | {r["support"]} | {percent(r["regex_f1"])} | {percent(r["before_f1"])} | {percent(r["after_f1"])} |' for r in per_labels)
    text.extend(['','## 属性附表','','| 字段 | 已知槽位 | 人工实体条件 | 端到端已知槽位 |','|---|---:|---:|---:|'])
    text.extend(f'| {r["field"]} | {r["support"]} | {percent(r["gold_accuracy"])} | {percent(r["end_to_end_accuracy"])} |' for r in queries['attributes']['rows'])
    text.extend(['','## 实验记录与边界','','- [数据准备清单](../../training_data/'+delivery_dir.name+'/preparation_audit.json)',
                 '- [训练回执](../../training/'+delivery_dir.name+'/experiment_receipt.json)',
                 '- [严格实体对比](../../evaluation/'+delivery_dir.name+'/entity_comparison.json)',
                 '- [模型耗时明细](../../training/'+delivery_dir.name+'/latency.json)',
                 '- [实际运行校验](../../evaluation/'+delivery_dir.name+'/runtime_verification.json)',
                 '- 代码记录了本轮未提交状态及输入、Schema、基础权重、最佳权重哈希。测试结果未用于本轮阈值或轮次选择。'])
    (delivery_dir/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (delivery_dir/'model_architecture.mmd').write_text('flowchart TD\n T[原始全文] --> W[384 token窗口 重叠96]\n W --> E[中文BERT 冻结前8层]\n E --> N[49类 多标签跨度实体]\n N --> S[起止 跨度均值 类型表示]\n S --> A[9个作用域属性头]\n E --> C[窗口CLS 病例四分类]\n C --> D[全文窗口logits均值]\n S --> R[实体对及中间语境关系头 仅已知正例监督]\n',encoding='utf-8')
    write_json(delivery_dir/'report_sources.json',{'data_dir':str(data_dir.resolve()),'training_dir':str(training_dir.resolve()),
               'rules':{'path':str(rules_path.resolve()),'sha256':sha256(rules_path)},
               'input_hashes':{str(p.resolve()):sha256(p) for p in [data_dir/'preparation_audit.json',data_dir/'split_manifest.json',training_dir/'experiment_receipt.json',training_dir/'after_metrics.json',training_dir/'latency.json']},
               'report_generator_sha256':sha256(__file__),'medical_gold':False})
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--delivery-dir',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--app-dir',type=Path)
    args=p.parse_args()
    build(args.data_dir,args.training_dir,args.delivery_dir,args.rules,args.app_dir)


if __name__=='__main__':
    main()
