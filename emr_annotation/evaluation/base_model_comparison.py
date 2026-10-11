"""Compare saved schema experiments without importing ML libraries or reading APIs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from emr_annotation.evaluation.entity_predictions import build_report, normalize_records
from emr_annotation.evaluation.schema_training_report import FIELD_NAMES, percent
from emr_annotation.training_data.schema_multitask import sha256, write_json


COMMON_PARAMETERS = ('epochs', 'seed', 'max_length', 'stride', 'max_span_width',
                     'batch_size', 'threshold', 'patience')
FROZEN_FILES = ('train.json', 'validation.json', 'test.json', 'reference.json',
                'schema.json', 'label_config.xml')


def validate_contract(baseline_data, candidate_data, baseline_receipt, candidate_receipt):
    checks = {}
    for name in FROZEN_FILES:
        first, second = sha256(baseline_data / name), sha256(candidate_data / name)
        if first != second:
            raise ValueError(f'Frozen input differs: {name}')
        checks[name] = first
    for name in COMMON_PARAMETERS:
        if baseline_receipt['parameters'][name] != candidate_receipt['parameters'][name]:
            raise ValueError(f'Comparison parameter differs: {name}')
    for receipt in (baseline_receipt, candidate_receipt):
        if not receipt.get('checkpoint_reloaded') or not receipt.get('all_checkpoint_tensors_finite'):
            raise ValueError('Both experiments must reload a finite saved checkpoint')
        if not receipt['status'].startswith('completed'):
            raise ValueError('Incomplete experiment')
    return checks


def build(baseline_data, baseline_training, candidate_data, candidate_training, output):
    read = lambda p: json.loads(Path(p).read_text('utf-8'))
    receipts = [read(p / 'experiment_receipt.json') for p in (baseline_training, candidate_training)]
    checks = validate_contract(baseline_data, candidate_data, *receipts)
    schema = read(baseline_data / 'schema.json')
    test = read(baseline_data / 'test.json')
    refs = [r for r in test if r['ner_supervised']]
    ids = {r['task_id'] for r in refs}
    labels = set(schema['labels'])
    norm = lambda rows: normalize_records([r for r in rows if r['task_id'] in ids], labels,
                                         reference=False, source='saved_text_only_prediction')
    predictions = [read(p / 'after_predictions.json') for p in (baseline_training, candidate_training)]
    for rows in predictions:
        if {r['task_id'] for r in rows} != {r['task_id'] for r in test} or len(rows) != len(test):
            raise ValueError('Predictions must cover every test task exactly once')
        expected = {r['task_id']: r['text'] for r in test}
        if any(r['text'] != expected[r['task_id']] for r in rows):
            raise ValueError('Prediction text differs from frozen test text')
    strict = build_report(normalize_records(refs, labels, reference=True, source='frozen_AI_sample'),
                          norm(predictions[0]), schema['controls'], after=norm(predictions[1]))
    metrics = [read(p / 'after_metrics.json') for p in (baseline_training, candidate_training)]
    runs = [strict['runs'][name] for name in ('before', 'after')]
    for measured, stored in zip(runs, metrics):
        if measured['overall'] != stored['entities']['runs']['before']['overall']:
            raise ValueError('Independent strict entity rescore differs from saved metrics')
    tokens = [read(p / 'token_audit.json') for p in (baseline_training, candidate_training)]
    latency = [read(p / 'latency.json') for p in (baseline_training, candidate_training)]
    names = ['中文BERT', 'mmBERT-base']
    scores, attrs, classes, entity_rows, token_rows, training_rows = [], [], [], [], [], []
    for name, m, r, t, speed in zip(names, metrics, receipts, tokens, latency):
        e = m['entities']['runs']['before']['overall']['micro']
        slots = sum(a['support'] for a in m['attributes'].values())
        correct = sum(a['correct'] for a in m['attributes'].values())
        conditional = sum(a['gold_entity_correct'] for a in m['attributes'].values())
        scores.append({'model': name, 'entity_precision': e['precision'], 'entity_recall': e['recall'],
                       'entity_f1': e['f1'], 'tp': e['tp'], 'fp': e['fp'], 'fn': e['fn'],
                       'entity_support': e['support'], 'attribute_support': slots,
                       'attribute_correct': correct, 'attribute_accuracy': correct / slots if slots else None,
                       'conditional_attribute_correct': conditional,
                       'conditional_attribute_accuracy': conditional / slots if slots else None,
                       'case_accuracy': m['case']['accuracy'], 'case_macro_f1': m['case']['macro_f1_supported'],
                       'target_recall': m['case']['per_class']['目标病例信号']['recall'],
                       'review_recall': m['case']['per_class']['待专业复核']['recall'],
                       'relation_support': m['relations']['known_positive_support'],
                       'relation_hits': m['relations']['known_positive_hits'],
                       'relation_recall': m['relations']['known_positive_recall'],
                       'p50_ms': speed['p50_ms'], 'p95_ms': speed['p95_ms'],
                       'documents_per_second': speed['documents_per_second']})
        training_rows.append({'model': name, 'parameters_total': r['parameters_total'],
                              'parameters_trainable': r['parameters_trainable'], 'best_epoch': r['best_epoch'],
                              'training_seconds': r['training_seconds'],
                              'peak_gpu_allocated_bytes': r['peak_gpu_allocated_bytes']})
        for field, a in m['attributes'].items():
            attrs.append({'model': name, 'attribute': FIELD_NAMES[field], 'support': a['support'],
                          'end_to_end': a['end_to_end_known_slot_accuracy'],
                          'conditional': a['gold_entity_known_slot_accuracy']})
        for label, c in m['case']['per_class'].items():
            classes.append({'model': name, 'class': label, **c})
        for split, audit in t.items():
            token_rows.append({'model': name, 'split': split, **audit})
    for label in schema['labels']:
        a, b = [r['per_label'][label] for r in runs]
        entity_rows.append({'label': label, 'support': a['support'], 'bert_f1': a['f1'],
                            'mmbert_f1': b['f1'], 'delta_pp': (b['f1'] - a['f1']) * 100
                            if a['f1'] is not None and b['f1'] is not None else None,
                            'bert_tp': a['tp'], 'bert_fp': a['fp'], 'bert_fn': a['fn'],
                            'mmbert_tp': b['tp'], 'mmbert_fp': b['fp'], 'mmbert_fn': b['fn']})
    a, b = scores
    delta = (b['entity_f1'] - a['entity_f1']) * 100
    title = f'mmBERT实体F1比中文BERT{"提高" if delta >= 0 else "降低"}{abs(delta):.2f}个百分点'
    if delta > 0 and b['case_accuracy'] < a['case_accuracy']:
        title = 'mmBERT提升了实体识别，但病例分类下降'
    summary = (f'同一批AI样例参考、同一患者划分下，mmBERT严格实体F1为{percent(b["entity_f1"])}，'
               f'中文BERT为{percent(a["entity_f1"])}，差值{delta:+.2f}个百分点。实体评分使用{len(refs)}例、'
               f'{a["entity_support"]}个参考实体；病例四分类使用{len(test)}例。\n\n'
               f'mmBERT病例准确率{percent(b["case_accuracy"])}，目标病例召回{percent(b["target_recall"])}；'
               f'中文BERT分别为{percent(a["case_accuracy"])}和{percent(a["target_recall"])}。'
               '本轮是固定训练方案、单一随机种子的基座替换试验；参考未获专家签署，开发批次已多次查看，不能据此认定医学泛化或显著优势。')
    summary += (f'\n\n已知属性端到端正确比例从{percent(a["attribute_accuracy"])}变为{percent(b["attribute_accuracy"])}；'
                '实体、属性与病例任务需分别判断，当前结果不足以支持整体替换。')
    sections = [
        {'id': 'construction', 'title': '一、建设思路回顾',
         'queryIds': ['training', 'scores'],
         'text': '## 一、建设思路回顾\n\n在已跑通的AI样例裁决、冻结、训练与严格评分流程上，将中文BERT编码器及官方分词器换为mmBERT-base，保留跨度实体、属性、病例和关系任务头。对照为上一轮真实训练并重载的中文BERT，不包含原版ModernBERT。两种模型是不同规模的预训练基座，本试验衡量固定方案下更换基座及分词器的整体效果。'},
        {'id': 'guide', 'title': '二、数据集标注指南研发过程', 'queryIds': ['training'],
         'text': '## 二、数据集标注指南研发过程\n\n沿用既有样例参考及同一XML快照：49类实体、9个实体属性、5种关系、病例四分类。指南和字典已另行补齐抗体IgG；本次没有重新裁决、改写原文或更换训练答案，也没有加入字符边界修复及规则融合。'},
        {'id': 'tools', 'title': '三、数据集工具与模型建设', 'queryIds': ['training', 'token_audit'],
         'text': '## 三、数据集工具与模型建设\n\n使用官方[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)，固定revision并校验权重SHA256。中文BERT有12层、mmBERT有22层，均只微调最后4层；嵌入及其他编码器模块冻结。学习率分别为编码器2e-5、任务头5e-4；batch 4、seed 42、阈值0.7、窗口384 token、重叠96 token、跨度32 token、最多12轮、patience 4。\n\n验证选模沿用实体F1、已知属性端到端正确比例和病例macro F1的等权均值。最佳权重实际保存并严格重载后生成测试预测。分词器不同，窗口数量、实体可表示性、有效监督及每轮更新次数会变化，报告分别列出；并非等参数量或等更新次数的架构消融。'},
        {'id': 'annotation', 'title': '四、标注情况与计划', 'queryIds': ['training', 'token_audit'],
         'text': '## 四、标注情况与计划\n\n两轮训练/验证/测试的JSON、全文、实体属性关系答案及XML逐文件字节一致；患者划分沿用128/27/27例。共182例AI裁决样例，33例实体监督被屏蔽，96个未决点仍待复核。关系只学习已确认样例正例，未知实体对不作为负例；属性只评价已知槽位。零支持类别保留不可评价。'},
        {'id': 'results', 'title': '五、初步结果（质量和速度）', 'queryIds': ['scores', 'labels', 'attributes', 'case_classes', 'token_audit'],
         'text': f'## 五、初步结果（质量和速度）\n\nmmBERT实体精确率{percent(b["entity_precision"])}、召回率{percent(b["entity_recall"])}、F1 {percent(b["entity_f1"])}；TP/FP/FN为{b["tp"]}/{b["fp"]}/{b["fn"]}。中文BERT对应为{percent(a["entity_precision"])}/{percent(a["entity_recall"])}/{percent(a["entity_f1"])}，TP/FP/FN为{a["tp"]}/{a["fp"]}/{a["fn"]}。\n\n'
                 f'已知属性端到端正确比例：mmBERT {b["attribute_correct"]}/{b["attribute_support"]}（{percent(b["attribute_accuracy"])}），中文BERT {a["attribute_correct"]}/{a["attribute_support"]}（{percent(a["attribute_accuracy"])}）。关系已知正例召回分别为{percent(b["relation_recall"])}与{percent(a["relation_recall"])}，完整关系精确率/F1仍不可评价。\n\n'
                 f'全部{len(test)}例测试样例的{tokens[0]["test"]["entities"]}个实体中，mmBERT不可表示{tokens[1]["test"]["unrepresentable_entities"]}个，中文BERT不可表示{tokens[0]["test"]["unrepresentable_entities"]}个；此分母与正式实体评分的{a["entity_support"]}个参考实体不同。分词可表示性改善不能单独解释F1差异。\n\n'
                 f'重载预热后的全文P50/P95：mmBERT {b["p50_ms"]:.2f}/{b["p95_ms"]:.2f}毫秒，中文BERT {a["p50_ms"]:.2f}/{a["p95_ms"]:.2f}毫秒。两次在同一本机分别测量一次，含分词、窗口编码、解码，不含网络和并发，不能作为稳定速度结论。'},
        {'id': 'next', 'title': '六、下一步计划', 'queryIds': ['scores', 'labels'],
         'text': '## 六、下一步计划\n\n依据逐类TP/FP/FN定位边界误差与语义混淆，再对字符边界修复或规则融合分别做对照。训练及阈值选择使用验证集，现有测试仅作为开发回归。补充专家确认、患者及近重复隔离的新病例作为独立测试，再进行多随机种子训练及重复测速。单纯更换基座的本轮结果不替代这些验证。'}
    ]
    for s in sections:
        s['sourceRowsByQuery'] = {}  # Filled with exactly the declared reviewed datasets below.
    datasets = {'scores': scores, 'labels': entity_rows, 'attributes': attrs,
                'case_classes': classes, 'token_audit': token_rows, 'training': training_rows}
    definitions = {
        'scores': '实体精确匹配原文字符起止及标签；属性要求实体跨度、标签和已知属性值匹配；病例按全文四分类；关系只计已知正例召回；速度为各运行一次全文串行测量。',
        'labels': '按标签列出全部严格TP/FP/FN；零参考支持类别F1为空，误报仍计入micro F1；差值为mmBERT减中文BERT的百分点。',
        'attributes': '仅已知槽位为分母；conditional为样例参考实体条件，end_to_end为预测实体条件，不是完整属性F1。',
        'case_classes': '病例四分类逐类P/R/F1；目标/待复核类别测试各仅4例。',
        'token_audit': '全部样例实体按各自官方分词器核查完整token边界、跨度长度和窗口覆盖；监督窗口额外受未决病例和不可表示实体屏蔽。',
        'training': '各运行真实保存重载回执记录的参数量、最佳轮次、训练时间及GPU分配峰值。'}
    sources = [baseline_training, candidate_training]
    queries = {}
    for key, rows in datasets.items():
        queries[key] = {'rows': rows, 'source': {'type': 'local_files',
            'files': [str(p / name) for p in sources for name in ('after_metrics.json', 'experiment_receipt.json', 'token_audit.json', 'latency.json')],
            'executedAt': receipts[1]['finished_at'],
            'evidenceFlow': [{'title': '冻结数据与真实模型对比', 'detail': '验证两轮冻结输入字节一致及公共参数相同；独立重算两组保存预测的严格实体指标，并与原回执评分逐项核对。其他任务沿用已保存统一评分。'}],
            'metricDefinitions': [{'label': key, 'definition': definitions[key],
                'componentIds': [s['id'] for s in sections if key in s['queryIds']] + [key + '-table']} ]},
            'methods': [{'language': 'python', 'code': 'python -m emr_annotation.evaluation.base_model_comparison --help; use the recorded baseline/candidate data and training directories with a new --output directory'}]}
    for s in sections:
        s['sourceRowsByQuery'] = {key: datasets[key] for key in s['queryIds']}
    now = datetime.now(timezone.utc).isoformat()
    snapshot = {'id': 'report:' + str(uuid.uuid4()), 'surface': 'report', 'title': title,
                'status': 'observed_ai_sample_not_expert_gold', 'generatedAt': now,
                'buildStatus': 'creating', 'filters': [],
                'report': {'summary': summary, 'asOf': receipts[1]['finished_at'][:10], 'sections': sections}, 'queries': queries}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'reviewed.json', snapshot)
    write_json(output / 'comparison.json', {'scores': scores, 'entity_f1_delta_pp': delta,
        'identical_frozen_input_hashes': checks, 'common_parameters': {k: receipts[0]['parameters'][k] for k in COMMON_PARAMETERS},
        'independent_entity_rescore': True, 'entity_comparison': strict,
        'source_files': {str(p / name): sha256(p / name) for p in sources for name in ('after_metrics.json', 'after_predictions.json', 'experiment_receipt.json')},
        'reference_status': 'ai_sample_not_expert_gold'})
    md = '# ' + title + '\n\n' + summary + '\n\n' + '\n\n'.join(s['text'] for s in sections)
    md += '\n\n## 逐实体附表\n\n|标签|参考数|中文BERT F1|mmBERT F1|差值（百分点）|\n|---|---:|---:|---:|---:|\n'
    for row in entity_rows:
        change = '不可评价' if row['delta_pp'] is None else f'{row["delta_pp"]:+.2f}'
        md += f'|{row["label"]}|{row["support"]}|{percent(row["bert_f1"])}|{percent(row["mmbert_f1"])}|{change}|\n'
    (output / 'report.md').write_text(md, encoding='utf-8')
    return snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-data', 'baseline-training', 'candidate-data', 'candidate-training', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = build(a.baseline_data, a.baseline_training, a.candidate_data, a.candidate_training, a.output)
    print(result['title'])


if __name__ == '__main__':
    main()
