"""Build a source-backed error-driven improvement report and complete read-only ledgers."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess
from .model_error_audit import inspect,render
from .explicit_evidence_binding import anatomical_conflict
from .entity_predictions import normalize_records,score_run
from .case_classification import case_metrics
from emr_annotation.training_data.schema_multitask import sha256,write_json
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(n):return f'{n*100:.2f}%'

def build(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);batch=args.batch_dir
    data=batch/'training_data/run_hybrid_002';freeze=batch/'adjudication/run_human_freeze_001'
    manifest=read(freeze/'freeze_manifest.json');checks={n:sha256(freeze/n)==v for n,v in manifest['outputs'].items()}
    if not all(checks.values()):raise ValueError('Frozen human reference changed')
    schema=read(data/'schema.json');train=read(data/'train.json');val=read(data/'validation.json');test=read(data/'test.json')
    baseline=batch/'evaluation/run_modular_002';combined=batch/'training/run_combined_evidence_001';reload=batch/'evaluation/run_combined_reload_001'
    fresh=read(reload/'predictions.json');expected=read(combined/'test_predictions.json');oldcase=read(baseline/'test_predictions.json')
    by={r['task_id']:r for r in expected};cb={r['task_id']:r for r in oldcase}
    reproduction=dict(cases=len(fresh),entities_exact=all(r['entities']==by[r['task_id']]['entities'] for r in fresh),relations_exact=all(r['relations']==by[r['task_id']]['relations'] for r in fresh),case_decisions_exact=all(r['case_decision']==cb[r['task_id']]['case_decision'] for r in fresh),case_probabilities_exact=all(r['case_probabilities']==cb[r['task_id']]['case_probabilities'] for r in fresh),actual_reload=read(reload/'reload_receipt.json'),frozen_files_unchanged=checks)
    if not all(reproduction[k] for k in ['entities_exact','relations_exact','case_decisions_exact','case_probabilities_exact']):raise ValueError('Actual reconstruction mismatch')
    write_json(out/'pipeline_verification.json',reproduction)
    bsum,berrors,bcases=inspect(test,read(baseline/'extractor_predictions.json'),schema,train,oldcase)
    asum,aerrors,acases=inspect(test,fresh,schema,train,fresh)
    for prefix,summary,errors,cases in [('before',bsum,berrors,bcases),('after',asum,aerrors,acases)]:
        write_json(out/(prefix+'_summary.json'),summary);write_json(out/(prefix+'_errors.json'),errors);write_json(out/(prefix+'_cases.json'),cases)
        (out/(prefix+'_review.html')).write_text(render(summary,errors,cases,'改进前' if prefix=='before' else '改进后'),encoding='utf-8')
    questions=[]
    for split,rows in [('train',[r for r in train if not r.get('synthetic')]),('validation',val),('test_development',test)]:
        for r in rows:
            for e in r['entities']:
                if anatomical_conflict(r['text'],e):questions.append(dict(split=split,task_id=r['task_id'],label=e['label'],start=e['start'],end=e['end'],mention=r['text'][e['start']:e['end']],context=r['text'][max(0,e['start']-65):e['end']+65],status='reference_question_not_modified'))
    write_json(out/'reference_questions.json',questions)
    queries={};sources=set()
    def query(key,rows,paths,definition,ids,formula=None):
        files=[dict(name=p.name,path=str(p.resolve()),sha256=sha256(p)) for p in paths];sources.update(paths)
        md=dict(label=key+'统计口径',definition=definition,componentIds=ids,sourceLineage=[dict(files=[str(p.resolve()) for p in paths])])
        if formula:md['formula']=formula
        queries[key]=dict(label=key,rows=rows,source=dict(kind='file',label='本批次冻结参考与实际训练/重载凭据',files=files,metricDefinitions=[md],reference_status='human_reviewed_not_expert_gold'),methods=[dict(language='python',code='Read the source JSON files; reproduce with python -m emr_annotation.evaluation.error_driven_report and report_manifest.json arguments. No source reference is rewritten.')])
    ner=[dict(model='现有mmBERT',model_short='现有模型',**bsum['entities']['overall']['micro'])]
    for rel,name,short in [('run_context_001','学习候选补全','候选补全'),('run_explicit_evidence_001','原文绑定与编号约束','原文约束'),('run_combined_evidence_001','候选补全与原文约束','组合方案')]:
        m=read(batch/('training/'+rel+'/test_metrics.json'))['entities']['overall']['micro'];ner.append(dict(model=name,model_short=short,**m))
    query('entities',ner,[baseline/'after_metrics.json']+[batch/('training/'+r+'/test_metrics.json') for r in ['run_context_001','run_explicit_evidence_001','run_combined_evidence_001']],
          '27例开发诊断、414个参考实体；实体字符起止及标签完全一致。严格micro F1=2TP/(2TP+FP+FN)。各分支按真实验证选择，设计受本批开发错误启发。',['report-summary','construction','results','entity-chart','entity-table'],'2*tp/(2*tp+fp+fn)')
    splits=[dict(split=split,real_cases=sum(not r.get('synthetic') for r in rows),synthetic_cases=sum(bool(r.get('synthetic')) for r in rows),patient_groups=len({str(r['patient_id']) for r in rows if not r.get('synthetic')})) for split,rows in [('训练',train),('验证',val),('开发测试',test)]]
    query('splits',splits,[data/(s+'.json') for s in ['train','validation','test']]+[freeze/'freeze_manifest.json'],'真实病例128/27/27；1000条合成样本只在训练。患者分组沿用冻结划分；未知标注不作负例。',['report-summary','guide','annotation','split-table'])
    basecfg=read(baseline/'model_config.json')
    options=schema['fields']['case_decision']['options']
    def case_metric(pred):
        byp={r['task_id']:r for r in pred}
        return case_metrics([options.index(r['case_decision']) for r in test],[options.index(byp[r['task_id']]['case_decision']) for r in test],options)
    cases=[dict(model='保留现有分层融合',validation_f1=basecfg['selected_case']['validation_macro_f1'],oof_f1=None,**case_metric(oldcase),selected=True)]
    for run,name in [('run_context_001','全文文本与冻结mmBERT'),('run_focused_case_001','聚焦文本与临床线索LR'),('run_clinical_cue_trees_002','临床线索小型树模型')]:
        p=batch/('training/'+run);cfg=read(p/'model_config.json');selected=cfg.get('selected',cfg.get('selected_case'))
        cases.append(dict(model=name,validation_f1=selected['validation']['macro_f1'],oof_f1=selected['oof']['macro_f1'],**case_metric(read(p/'test_case_predictions.json')),selected=False))
    query('cases',cases,[baseline/'model_config.json',baseline/'test_predictions.json',data/'test.json']+[batch/('training/'+r+'/'+name) for r in ['run_context_001','run_focused_case_001','run_clinical_cue_trees_002'] for name in ['model_config.json','test_case_predictions.json','test_metrics.json']],
          '病例四分类macro F1为四类F1算术平均；测试27例，目标与复核各4例。新小模型按患者分组3折OOF与验证F1均值选内部配置；验证未超过现有0.60，不替换现有分类器。不同OOF仅在新小模型间可比较。',['report-summary','tools','results','case-chart','case-table','next'])
    counts=Counter(r['task_id'] for r in berrors if r['domain']=='entity');top=counts.most_common(6)
    causes=[dict(observation='误报以数值/体温为主',count=sum(r['direction']=='fp' and r['label'] in ['数值','体温'] for r in berrors if r['domain']=='entity'),denominator=113,status='直接核查',intervention='学习候选判别；排除明确椎体编号'),
            dict(observation='时间表达漏报',count=sum(r['direction']=='fn' and r['label']=='时间表达' for r in berrors if r['domain']=='entity'),denominator=88,status='直接核查',intervention='日期/时长候选与局部上下文判别'),
            dict(observation='阳性误为阴性',count=6,denominator=7,status='直接核查',intervention='病原体同条记录结果绑定，排除参考区间'),
            dict(observation='错误最多病例的实体错误',count=top[0][1],denominator=201,status='直接核查',intervention='逐点病例复核；不改人工参考'),
            dict(observation='模板合成与真实病例存在差异',count=None,denominator=None,status='原因假设，尚无单因子证明',intervention='后续真实少数类、模板分层消融与折外特征')]
    query('causes',causes,[out/'before_errors.json',data/'train.json'],'FP与FN各自有分母，不能相加为病例数；样本缺少、上下文和上游共现仅是待验证原因。',['construction','tools','results','cause-table','next'])
    attrs=[];values=[]
    labels={'test_result':'检验结果','specimen_type':'标本类型','detection_method':'检测方式','temporality':'时间属性','experiencer':'主体','finding_context':'存在状态','clinical_course':'病程变化','conditional':'条件','hypothetical':'假设'}
    for f in [f for f,c in schema['fields'].items() if c['per_region']]:
        before=bsum['attributes'].get(f,{});after=asum['attributes'].get(f,{})
        attrs.append(dict(attribute=labels.get(f,f),support=before.get('support',0),before=before.get('correct'),after=after.get('correct'),upstream_errors=after.get('upstream_entity_error'),matched_errors=after.get('matched_attribute_error')))
    for f in ['test_result','specimen_type','detection_method']:
        for choice in asum['pathogen_attributes']['fields'][f]['choices']:
            old=next(r for r in bsum['pathogen_attributes']['fields'][f]['choices'] if r['option']==choice['option'])
            values.append(dict(field=labels[f],option={'positive':'阳性','negative':'阴性','unknown':'无法判断'}.get(choice['option'],choice['option']),support=choice['support'],before=old['correct'],after=choice['correct'],recall=choice['known_slot_recall']))
    query('attributes',attrs,[out/'before_summary.json',out/'after_summary.json'],'仅在有值的参考槽位评分；上游实体不匹配算错误；无参考病程属性不可评价。',['guide','results','attribute-table'])
    query('detection',values,[out/'before_summary.json',out/'after_summary.json'],'病原检测已知槽位召回，分母是每个值的真实参考数；无参考为null。不是完整新预测记录精确率。',['results','detection-table'])
    lab=[dict(label=k,support=v['support'],before_f1=bsum['entities']['per_label'][k]['f1'],after_f1=v['f1'],tp=v['tp'],fp=v['fp'],fn=v['fn']) for k,v in asum['entities']['per_label'].items()]
    query('labels',lab,[out/'before_summary.json',out/'after_summary.json'],'49类全部保留；无参考类F1为空，不作为已达标能力。',['results','label-table'])
    bm=bsum['entities']['overall']['micro'];am=asum['entities']['overall']['micro'];cm=cases[0];path=asum['pathogen_attributes'];rel=asum['relations']
    positive_patients=len({str(r['patient_id']) for r in test if any(e.get('attributes',{}).get('test_result')=='positive' for e in r['entities'])})
    summary=f"严格实体F1由{pct(bm['f1'])}提高到{pct(am['f1'])}，增加{(am['f1']-bm['f1'])*100:.2f}个百分点；误报{bm['fp']}→{am['fp']}、漏报{bm['fn']}→{am['fn']}。\n\n检验阳性识别1/7→7/7（来自{positive_patients}位患者），同条记录绑定修复6处阳性误判；标本类型仅1处具体参考，尚不能证明多标本识别有效。\n\n病例分类保留现有分层融合，macro F1为{pct(cm['macro_f1'])}；小模型试验最高开发分数65.42%，验证不足以支持替换。80%—85%的双目标尚未完成。以上是已参与开发的27例诊断结果。"
    sections=[
        dict(id='construction',title='一、建设思路回顾',queryIds=['entities','causes'],text='## 一、建设思路回顾\n\n本轮先建立逐项错分账本，再对主要错误做对照。95/113个实体误报属于数值或体温；35/88个漏报是时间表达。单一病例贡献96/201个实体错误，多处文本是椎体编号。\n\n模型共用现有mmBERT跨度抽取器：以学习型局部候选补全提高召回，以明确原文和标签语义约束减少误报。病例分类继续与实体模型分开比较。编号过滤同时删除7个参考实体，已保留疑点，人工参考未改写。'),
        dict(id='guide',title='二、数据集标注指南研发过程',queryIds=['splits','attributes'],text='## 二、数据集标注指南研发过程\n\n沿用冻结时的49类实体、9个实体关联属性、5类关系与病例四分类。病原体、检测结果、标本和方法均按同条检测记录关联；参考区间中的阴性不作为本次结果，缺失标注也不当阴性。\n\n已知属性错误进一步分为实体没有抽到和实体抽对但属性判断错误。具体标本、少见检测方式与病程属性的真实参考仍稀少，不能用多数值的高准确率代替覆盖能力。'),
        dict(id='tools',title='三、数据集工具与模型建设',queryIds=['cases','causes'],text='## 三、数据集工具与模型建设\n\n实际训练并重载了局部字符上下文候选头、实体查询注意力属性头，以及全文/聚焦文本、冻结mmBERT和临床线索树模型。注意力头验证未优于原头，保留epoch 0；它未修复阳性问题。原文绑定是确定性约束，不能把它的收益写成新神经网络训练收益。\n\n候选补全仅由训练拟合、验证选阈值；组合方案也先保存验证选择再读开发测试参考。当前病例小模型均未超过原验证macro F1 0.60，保持现有分层融合。新模型用患者分组3折OOF评估，冻结编码器不使用病例标签；原融合训练特征仍非完整折外抽取。\n\n[临床实体与断言联合抽取论文](https://aclanthology.org/P19-1091/)支持共享表示和联合任务的研究方向；[GlobalPointer论文](https://arxiv.org/abs/2208.03054)提供跨度建模与不平衡损失思路，本轮未实施其原损失。[官方mmBERT集合](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder)本次核查可见small/base，未确认官方large权重，继续使用base。没有和原版ModernBERT比较。'),
        dict(id='annotation',title='四、标注情况与计划',queryIds=['splits'],text='## 四、标注情况与计划\n\n182例、11,845点人工审查已冻结，全部四个冻结文件哈希本次核对一致。保留审查人决定、历史版本与原划分，未宣称独立专家金标。真实训练/验证/开发测试为128/27/27，1000条规则合成数据只用于训练；新增病例小模型对照使用128例真实训练，基础抽取器和属性头保留合成数据训练来源。\n\n所有原文或标签冲突仅写入独立疑点清单；本轮不修改参考答案。关系参考主要为正例，不输出未经支持的完整精确率和F1。'),
        dict(id='results',title='五、初步结果',queryIds=['entities','cases','attributes','detection','labels'],text=f"## 五、初步结果\n\n组合实体精确率{pct(am['precision'])}、召回率{pct(am['recall'])}，严格F1 {pct(am['f1'])}。候选补全单独引入较多误报；原文约束单独召回下降；组合后净增24个正确实体，但剩余64个漏报和84个误报需要继续处理。\n\n检验结果115/117正确；阳性7/7、阴性98/100、无法判断10/10，余下2处是病原实体缺失。标本114/117、方式114/116，三项完整已知记录114/116；具体血液标本1处未识别，各类标本能力不能据此验收。\n\n病例四分类仍为{pct(cm['macro_f1'])}，8/27错分；目标病例与待复核各仅4个开发参考。聚焦文本LR试验为65.42%、树模型64.50%，按验证规则均未替代现有分类器。模型规模和分类头复杂度尚未证明能解决真实少数类和判断边界不足。\n\n已知关系正例召回{pct(rel['recall'])}（{rel['strict_hits']}/{rel['known_positive_support']}）；与病例病因判断分开。实际重建27例实体、关系、病例类别和概率全部一致；204项离线测试通过。"),
        dict(id='next',title='六、下一步计划',queryIds=['causes','cases'],text='## 六、下一步计划\n\n下一轮优先复核椎体编号等参考疑点，并补充“目标病例/待专业复核”真实病例和邻近易混负例。阳性7处来自少量病例，需要补充不同标本、方式、时期、同病例多次结果的真实关联参考。\n\n病例改进重点转向按患者完整折外抽取特征、临床证据的时间/否定组合以及少数类样本，而非继续只扩大模型或增加随机参数试验。对合成数据真实匹配度做固定患者折的分层消融；保留四分类目标，不以二分类得分代替。\n\n最后用未参与本轮设计的新真实批次验收严格实体micro F1与病例macro F1各80%—85%。当前目标保留，未标完成；历史报告、人工审查工具和线上Label Studio服务保留。')]
    next(s for s in sections if s['id']=='tools')['text']+='\n\n保留的病例分支有独立的专用抽取器和分类权重；新实体纠错并未自动进入其训练特征。本轮病例新试验读原文特征，尚未完成基于纠错实体的患者完整折外融合重训。'
    snapshot=dict(surface='report',title=f"实体F1达{pct(am['f1'])}，病例分类待提升",status='human_reviewed_development_diagnostic',generatedAt=datetime.now(timezone.utc).isoformat(),buildStatus='complete',filters=[],report=dict(asOf='2026-10-12',summary=summary,sections=sections),queries=queries)
    write_json(out/'reviewed_snapshot.json',snapshot)
    (out/'report.md').write_text('# '+snapshot['title']+'\n\n'+summary+'\n\n'+'\n\n'.join(s['text'] for s in sections),encoding='utf-8')
    sources.update([freeze/'freeze_manifest.json',reload/'reload_receipt.json',combined/'experiment_receipt.json'])
    write_json(out/'report_manifest.json',dict(stage='delivery',created_at=snapshot['generatedAt'],source_sha256={str(p.resolve()):sha256(p) for p in sources},inputs=dict(batch_dir=str(batch.resolve()),output_dir=str(out.resolve())),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_tree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),code_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))},goal_status='not_achieved_case_macro_f1_below_80_percent',outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print('ERROR_DRIVEN_REPORT',out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['batch-dir','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    build(p.parse_args())
if __name__=='__main__':main()
