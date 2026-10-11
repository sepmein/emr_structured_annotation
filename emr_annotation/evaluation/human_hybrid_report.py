"""Build the six-section human-reviewed mmBERT report from saved receipts.

Stdlib only. Contains no patient records; all snapshots are written to output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from emr_annotation.training_data.schema_multitask import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def percent(value):
    return '不可评价' if value is None else f'{value*100:.2f}%'


def build(data_dir, freeze_dir, extractor_dir, fusion_dir, output_dir, app_dir=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = read(data_dir/'preparation_audit.json')
    freeze = read(freeze_dir/'freeze_manifest.json')
    reference = read(freeze_dir/'reference.json')
    queries = {}

    def query(name, rows, paths, definition, components):
        queries[name] = dict(rows=rows, source=dict(type='local_files',
            files=[str(p.resolve()) for p in paths], sha256={str(p.resolve()):sha256(p) for p in paths},
            executedAt=datetime.now(timezone.utc).isoformat(),
            evidenceFlow=[dict(title='已保存运行证据', detail=definition)],
            metricDefinitions=[dict(label=name, definition=definition, componentIds=components)]))

    splits = [dict(split=n, real=audit['real_split_counts'][n],
                   synthetic=audit['synthetic_cases'] if n=='train' else 0,
                   total=audit['split_counts'][n]) for n in ('train','validation','test')]
    query('splits', splits, [data_dir/'preparation_audit.json',freeze_dir/'freeze_manifest.json'],
          '病例数按集合及来源分开。患者划分沿用既有128/27/27；1000条自动规则样本只用于训练。',
          ['construction','guide','annotation','splits-table','report-summary'])
    coverage = [dict(label=k, human=v['real'], synthetic=v['synthetic']) for k,v in audit['label_coverage'].items()]
    query('coverage', coverage, [data_dir/'preparation_audit.json'],
          '全量人工参考与生成样本的实体跨度数；合成覆盖不代表真实测试性能。', ['guide','coverage-table'])
    cases = [dict(label=k, count=v) for k,v in Counter(r['case_decision'] for r in reference).items()]
    query('human_cases', cases, [freeze_dir/'reference.json'],
          '182例最终人工确认的病例四分类分布，非专家签署状态保留。', ['annotation','human-cases-table'])

    completed = fusion_dir and (fusion_dir/'comparison.json').is_file()
    scores, classes, auxiliary, labels, attributes, training = [], [], [], [], [], []
    summary = f"{freeze['cases']}例人工裁决已完整冻结，{freeze['points']}个审查点均已处理。1000条规则生成样本已覆盖49类实体、5种关系，全部进入训练集。mmBERT抽取与融合训练正在进行，当前尚无最终效果结论。"
    result_text = '训练正在进行。最终比较将使用27例真实病例，文本、结构化证据和融合分支保持相同测试集合。'
    title = '人工裁决已冻结，mmBERT混合模型正在训练'
    sources = [data_dir/'preparation_audit.json']
    if completed:
        comparison = read(fusion_dir/'comparison.json')
        extraction = read(extractor_dir/'after_metrics.json')
        er = next(iter(extraction['entities']['runs'].values()))
        names = {'text':'文本分支','evidence':'结构化证据分支','fusion':'混合判断分支','extractor_text_case':'原抽取器文本分类头'}
        for branch,m in comparison.items():
            c=m['case']; target=next(x for x in c['per_class'] if x['label']=='目标病例信号')
            review=next(x for x in c['per_class'] if x['label']=='待专业复核')
            scores.append(dict(branch=branch,model=names[branch],support=c['support'],accuracy=c['accuracy'],macro_f1=c['macro_f1'],target_recall=target['recall'],review_recall=review['recall']))
            classes += [dict(model=names[branch],**p) for p in c['per_class']]
            for kind in ('syndrome','pathogen'):
                if kind in m:
                    auxiliary += [dict(model=names[branch],kind='症候群证据分组' if kind=='syndrome' else '病原检出分组',**p) for p in m[kind]['per_label']]
        labels = [dict(label=k,**v) for k,v in er['per_label'].items()]
        attributes = [dict(attribute=k,**v) for k,v in extraction['attributes'].items()]
        receipt=read(extractor_dir/'experiment_receipt.json'); fr=read(fusion_dir/'experiment_receipt.json')
        training=[dict(stage='mmBERT抽取',best_epoch=receipt['best_epoch'],validation_selection=receipt['best_validation_score'],checkpoint_reloaded=receipt['checkpoint_reloaded']),
                  *[dict(stage=names[b],best_epoch=fr['best_epochs'][b],validation_selection=fr['best_validation_case_macro_f1'][b],checkpoint_reloaded=fr['checkpoint_reload_exact'][b]) for b in ('text','evidence','fusion')]]
        f=comparison['fusion']; t=comparison['text']; difference=(f['case']['accuracy']-t['case']['accuracy'])*100
        f_target=next(p for p in f['case']['per_class'] if p['label']=='目标病例信号')
        t_target=next(p for p in t['case']['per_class'] if p['label']=='目标病例信号')
        f_review=next(p for p in f['case']['per_class'] if p['label']=='待专业复核')
        t_review=next(p for p in t['case']['per_class'] if p['label']=='待专业复核')
        f_correct=round(f['case']['accuracy']*f['case']['support'])
        t_correct=round(t['case']['accuracy']*t['case']['support'])
        micro=er['overall']['micro']; rel=extraction['relations']
        title = 'mmBERT混合模型已训练，效果以真实病例比较为准'
        summary = (f"{freeze['cases']}例人工裁决已冻结；1000条规则生成样本只用于训练。27例真实测试病例上，混合判断准确率{percent(f['case']['accuracy'])}、宏平均F1为{percent(f['case']['macro_f1'])}；文本分支分别为{percent(t['case']['accuracy'])}、{percent(t['case']['macro_f1'])}，准确率差值{difference:+.2f}个百分点。\n\n"
                   f"目标病例识别从文本分支{t_target['tp']}/{t_target['support']}到混合分支{f_target['tp']}/{f_target['support']}；但待专业复核识别从{t_review['tp']}/{t_review['support']}变为{f_review['tp']}/{f_review['support']}。严格实体F1为{percent(micro['f1'])}，参考实体{micro['support']}个。小样本结果只能作为本批次试验观察。症候群和病原输出来自可追溯证据分组，尚未得到独立专家标注；病原缺失结果按未知处理。")
        result_text = (f"三个分支共用同一mmBERT抽取器，以实际预测实体、属性和关系生成证据特征，验证集分别选模。混合判断准确率{percent(f['case']['accuracy'])}，文本分支{percent(t['case']['accuracy'])}；差值{difference:+.2f}个百分点。比较针对分支输入与融合方式，没有重新训练三个独立编码器。\n\n"
                       f"混合分支正确{f_correct}/{f['case']['support']}，文本分支{t_correct}/{t['case']['support']}，正确数差值{f_correct-t_correct:+d}例。目标病例召回从{percent(t_target['recall'])}（{t_target['tp']}/{t_target['support']}）到{percent(f_target['recall'])}（{f_target['tp']}/{f_target['support']}）；待专业复核召回从{percent(t_review['recall'])}（{t_review['tp']}/{t_review['support']}）变为{percent(f_review['recall'])}（{f_review['tp']}/{f_review['support']}）。原抽取器文本分类头准确率为{percent(comparison['extractor_text_case']['case']['accuracy'])}，宏平均F1为{percent(comparison['extractor_text_case']['case']['macro_f1'])}，三者的类别平衡不同。\n\n"
                       f"实体按字符起止位置及标签完全一致计分：精确率{percent(micro['precision'])}、召回率{percent(micro['recall'])}、F1 {percent(micro['f1'])}。{er['overall']['supported_label_count']}类在测试中有参考，其余类别保留不可评价；合成样本覆盖49类不能替代真实验证。\n\n"
                       f"已知关系正例召回{percent(rel['known_positive_recall'])}（{rel['known_positive_hits']}/{rel['known_positive_support']}）。真实病例未全面标注无关系负例，因此不报告完整关系精确率和F1。合成记录的合法无关系配对用于负例训练。\n\n"
                       f"混合分支症候群证据micro F1为{percent(f['syndrome']['micro_f1'])}；病原检出micro F1为{percent(f['pathogen']['micro_f1'])}，仅有{f['pathogen']['known_slots']}个已知槽位、{f['pathogen']['positive_support']}个阳性参考，能力仍弱。症候群结果衡量与人工实体属性派生证据标签的一致性；呼吸受损和严重全身表现没有真实阳性参考，细菌、真菌也没有阳性支持，不能宣称这些分类有效。没有检验结果的分组不进入病原分母。检出分组不代表肺炎病因确认。")
        sources=[fusion_dir/'comparison.json',fusion_dir/'experiment_receipt.json',extractor_dir/'after_metrics.json',extractor_dir/'experiment_receipt.json']
    query('scores',scores,sources,'病例准确率=正确数/27；macro F1按有参考的四类平均。目标、复核召回分别以该类真实参考数为分母。', ['report-summary','results','scores-table','case-chart','next'])
    query('case_classes',classes,sources,'每分支逐类别TP/FP/FN与精确率、召回率、F1。', ['results','case-classes-table'])
    query('auxiliary',auxiliary,sources,'症候群是人工实体属性派生分组；病原仅在结果已知槽位上评分。未观察标签不当成阴性。', ['results','auxiliary-table'])
    query('entity_labels',labels,sources,'严格字符跨度与标签匹配；无参考类别F1为空，仍保留误报。', ['results','entity-labels-table'])
    query('attributes',attributes,sources,'端到端要求实体跨度、标签及已知属性匹配；gold实体条件结果仅用于定位错误，不是推理输入。', ['results','attributes-table'])
    query('training',training,sources,'抽取器使用验证集实体、属性、病例三指标均值；融合各分支使用真实验证病例macro F1选模。', ['tools','training-table'])
    sections = [
        ('construction','一、建设思路回顾',['splits'],
         '从已完成的人工审查结果继续训练。模型由mmBERT文本编码、跨度实体与属性预测、关系预测、结构化证据整理和门控融合组成。病例输出保留目标病例信号、待专业复核、非目标、信息不足；增加发热、肺炎、呼吸受损、严重全身表现四个证据分组，以及病毒、细菌、真菌、支原体或衣原体四个检出分组。'),
        ('guide','二、数据集标注指南研发过程',['splits','coverage'],
         '本轮使用审查时的XML快照：49类实体、9个属性、5种关系和病例四分类。否定、既往、假设、条件和非患者主体不作为当前症状证据；结果未知不等于阴性，测量必须有明确的指标→数值关系。测量字段没有症状断言属性，数值特征排除明确既往、计划和他人上下文，未标时间保留为未指定。\n\n'
         '1000条样本按裁决规范生成，记录逐实体、逐关系和病例规则依据。覆盖儿童线索、少见病原、起始/结束时间、既往已缓解、家属经历、条件句、计划检查、阴性和信息不足。它们是自动规则裁决的合成训练数据，没有逐条人工或专家签署；原182例人工裁决保留不变。'),
        ('tools','三、数据集工具与模型建设',['training','scores'],
         '使用官方[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)的本地权重与分词器。抽取器微调最后4层及实体、属性、关系、病例头；窗口384 token、重叠96、跨度32，batch 4，seed 42，实体阈值0.7。融合器读取全文窗口平均CLS、文本病例概率，以及预测实体断言、检验结果、关系和显式数值。\n\n'
         '两阶段训练：先训练抽取器，再冻结抽取器训练文本、证据及门控融合分支。每个分支都有病例、症候群和病原头；真实样本损失权重5，合成样本1。辅助二值阈值固定0.5。训练特征由抽取器对训练记录本身预测得到，本轮没有交叉折外抽取；测试推理只读task_id与text，不读取人工答案。\n\n'
         '权重保存后严格重载，并记录输入/代码/权重哈希、环境和选择依据。E盘容量不足，本轮权重与报告实际保存在D盘，E盘output保留Windows Junction入口；备份必须同时保留实际目标目录。Label Studio在线服务尚未接入混合分支。'),
        ('annotation','四、标注情况与计划',['splits','human_cases'],
         f"全部{freeze['cases']}例、{freeze['points']}个审查点完成确认，数据库历史和最终导出已有独立冻结副本。真实128/27/27例训练、验证、测试沿用原患者划分；1000条合成数据仅加入训练，训练共1128例。人工确认记录仍标为未获独立专家签署。测试批次过去曾用于开发与查看，本轮是开发集内比较，需要新批次才能验证泛化。"),
        ('results','五、初步结果',['scores','case_classes','auxiliary','entity_labels','attributes'],result_text),
        ('next','六、下一步计划',['scores'],
         '先针对真实测试误判检查肺炎断言、边界、检验结果和关系漏报，再扩充人工真实病例。对症候群与病因另建专家确认目标，补充真实关系负例，使用患者分组交叉折外抽取特征训练融合器，最后在未参与开发的新批次评估。当前完成训练和离线预测，在线Label Studio接入及医学验收需要后续独立实施。')]
    stable_id='report:human-mmbert-fusion-20261011-001'
    if app_dir and (app_dir/'src/data.json').exists():
        stable_id=read(app_dir/'src/data.json')['id']
    snapshot=dict(id=stable_id,surface='report',title=title,status='human_reviewed_with_synthetic_training_not_expert_gold',
                  generatedAt=datetime.now(timezone.utc).isoformat(),buildStatus='complete' if completed else 'creating',filters=[],
                  report=dict(summary=summary,asOf='2026-10-11',sections=[dict(id=i,title=t,queryIds=q,text=f'## {t}\n\n{x}') for i,t,q,x in sections]),queries=queries)
    write_json(output_dir/'reviewed_snapshot.json',snapshot)
    if app_dir:
        write_json(app_dir/'src/data.json',snapshot)
    table='| 分支 | 测试数 | 病例准确率 | 宏平均F1 | 目标召回 |\n|---|---:|---:|---:|---:|\n'
    table+='\n'.join(f"| {r['model']} | {r['support']} | {percent(r['accuracy'])} | {percent(r['macro_f1'])} | {percent(r['target_recall'])} |" for r in scores)
    narrative='# '+title+'\n\n'+summary+'\n\n'+'\n\n'.join(s['text']+(('\n\n'+table) if s['id']=='results' else '') for s in snapshot['report']['sections'])
    (output_dir/'report.md').write_text(narrative+'\n',encoding='utf-8')
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data-dir','freeze-dir','extractor-dir','output-dir'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fusion-dir',type=Path);p.add_argument('--app-dir',type=Path)
    a=p.parse_args();snapshot=build(a.data_dir,a.freeze_dir,a.extractor_dir,a.fusion_dir,a.output_dir,a.app_dir)
    print(json.dumps(dict(title=snapshot['title'],buildStatus=snapshot['buildStatus']),ensure_ascii=False))


if __name__=='__main__':main()
