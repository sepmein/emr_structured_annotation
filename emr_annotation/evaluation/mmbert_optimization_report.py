"""Six-part optimization report from actual validation/evaluation receipts."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from emr_annotation.training_data.schema_multitask import sha256,write_json


def read(p):return json.loads(Path(p).read_text('utf-8'))
def pct(v):return '不可评价' if v is None else f'{v*100:.2f}%'


def build(args,complete=False):
    root=args.batch_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data=root/'training_data/run_hybrid_002';base=root/'training/run_hybrid_003';baseline=root/'training/run_hybrid_fusion_001'
    evaluation=args.evaluation_dir;ecfg=read(evaluation/'model_config.json');extractor=Path(ecfg['ner_extractor_path'])
    er=read(extractor/'experiment_receipt.json');ex=read(evaluation/'after_metrics.json');micro=ex['entities']['runs']['before']['overall']['micro']
    oldmicro=read(base/'after_metrics.json')['entities']['runs']['before']['overall']['micro']
    comparison=read(evaluation/'comparison.json');oldcase=read(baseline/'comparison.json')['fusion']['case']
    case=comparison['selected']['case'];queries={}
    def query(key,rows,paths,formula,ids):
        files=[dict(name=str(p.resolve()),path=str(p.resolve()),sha256=sha256(p)) for p in paths]
        definition=dict(label=key+'统计口径',definition=formula,componentIds=ids,
                        sourceLineage=[dict(files=[f['name'] for f in files])])
        if key=='entity_comparison':definition['formula']='2*tp/(2*tp+fp+fn)'
        if key=='scores':definition['formula']='mean(F1 of the four case classes)'
        queries[key]=dict(label=key,rows=rows,source=dict(kind='file',label='本批次实际运行与冻结数据',
            files=files,metricDefinitions=[definition],reference_status='human_reviewed_not_expert_gold'),
            methods=[dict(language='python',code='Read the listed UTF-8 JSON files and assemble this query using build() in emr_annotation/evaluation/mmbert_optimization_report.py. The exact CLI arguments are recorded in report_manifest.json.')])
    query('entity_comparison',[dict(model='原mmBERT抽取器',**oldmicro),dict(model='验证选中的改进抽取器',**micro)],
          [base/'after_metrics.json',evaluation/'after_metrics.json'],'实体字符起止及标签完全一致；micro F1=2TP/(2TP+FP+FN)，参考414个。',
          ['report-summary','results','entity-comparison-table','entity-chart'])
    earlier=root/'evaluation/run_modular_001/after_metrics.json'
    if earlier.exists():
        prior=read(earlier)['entities']['runs']['before']['overall']['micro']
        queries['entity_comparison']['rows'].insert(1,dict(model='第一轮修正抽取器',**prior))
        file=dict(name=str(earlier.resolve()),path=str(earlier.resolve()),sha256=sha256(earlier))
        queries['entity_comparison']['source']['files'].append(file)
        queries['entity_comparison']['source']['metricDefinitions'][0]['sourceLineage'][0]['files'].append(file['name'])
    scores=[dict(model='原mmBERT融合分类器',branch='prior_fusion',selected=False,**oldcase)]
    names={'neural_text':'分层神经文本','neural_evidence':'分层神经证据','neural_fusion':'分层神经融合',
           'small_text':'逻辑回归：字词文本','small_embedding':'逻辑回归：mmBERT文本向量',
           'small_evidence':'逻辑回归：结构化证据','small_fusion':'逻辑回归：字词＋证据',
           'small_semantic_fusion':'逻辑回归：文本向量＋证据','small_xgboost':'浅层XGBoost：结构化证据'}
    selected_key=ecfg['selected_case']['family']+'_'+ecfg['selected_case']['branch']
    for b,m in comparison.items():
        if b=='selected':continue
        scores.append(dict(model=names[b],branch=b,selected=b==selected_key,**m['case']))
    for row in scores:
        row['target_recall']=next(p['recall'] for p in row['per_class'] if p['label']=='目标病例信号')
        row['review_recall']=next(p['recall'] for p in row['per_class'] if p['label']=='待专业复核')
    query('scores',scores,[evaluation/'comparison.json',evaluation/'model_config.json',baseline/'comparison.json'],
          '同一27例真实开发测试；病例macro F1为四类F1平均。选中标记依据验证集，而非本表测试结果。',
          ['report-summary','results','scores-table','next'])
    query('case_classes',[dict(model=r['model'],**p) for r in scores for p in r['per_class']],
          [evaluation/'comparison.json',baseline/'comparison.json'],'逐类别TP/FP/FN与支持数。', ['results','case-classes-table'])
    validation=[]
    for name in ('run_hybrid_fusion_001','run_hierarchical_001','run_hierarchical_002','run_hierarchical_003'):
        p=root/'training'/name/'experiment_receipt.json';r=read(p)
        for b,score in r['best_validation_case_macro_f1'].items():validation.append(dict(run=name,branch=b,macro_f1=score))
    small=Path(ecfg['small_path']);candidates=read(small/'validation_candidates.json')
    validation += [dict(run=small.name,branch=c['candidate'],macro_f1=c['validation']['macro_f1']) for c in candidates]
    query('validation',validation,[small/'validation_candidates.json']+[root/'training'/n/'experiment_receipt.json' for n in ('run_hybrid_fusion_001','run_hierarchical_001','run_hierarchical_002','run_hierarchical_003')],
          '所有分数来自27例真实验证；合成样本不进入此分母；小验证集反复选模存在乐观偏差。', ['tools','validation-table'])
    preparation=read(data/'preparation_audit.json')
    splits=[dict(split=s,real=preparation['real_split_counts'][s],synthetic=1000 if s=='train' else 0,total=preparation['real_split_counts'][s]+(1000 if s=='train' else 0)) for s in ('train','validation','test')]
    query('splits',splits,[data/'preparation_audit.json'],'真实患者分组128/27/27，1000条合成仅加入训练。',['construction','annotation','splits-table','report-summary'])
    schema=read(data/'schema.json');training=read(data/'train.json');attribute_training=[]
    for synthetic in (False,True):
        for field in ('test_result','specimen_type','detection_method'):
            counts=Counter(e['attributes'][field] for r in training if bool(r.get('synthetic'))==synthetic for e in r['entities'] if field in e.get('attributes',{}))
            for option,count in counts.items():
                attribute_training.append(dict(source='合成' if synthetic else '真实',field=field,option=schema['fields'][field]['values'][option],support=count))
    query('attribute_training',attribute_training,[data/'train.json',data/'schema.json'],
          '只计已标注的训练实体属性；缺失不填阴性；真实128例与合成1000条分别汇总。',['guide','attribute-training-table'])
    audit=[]
    for label,path in [('原抽取器',base),('改进抽取器',extractor)]:
        p=path/'token_audit.json'
        if not p.exists():p=path/'tokenizer_alignment.json'
        if p.exists():
            obj=read(p)
            for s,v in obj.items():
                if isinstance(v,dict):audit.append(dict(model=label,split=s,**v))
    # Canonical training receipts carry the alignment audit even when no sidecar is written.
    for label,path in [('原抽取器',base),('改进抽取器',extractor)]:
        r=read(path/'experiment_receipt.json')
        for s,v in r.get('token_audit',{}).items():audit.append(dict(model=label,split=s,**v))
    query('alignment',audit,[base/'token_audit.json',extractor/'token_audit.json'],
          '训练窗口的可表示性审计；覆盖不代表预测正确；改进使用token包围范围和字符边界头。',['tools','alignment-table'])
    query('attributes',[dict(attribute=k,**v) for k,v in ex['attributes'].items()],[evaluation/'after_metrics.json'],
          '端到端已知属性槽位要求实体匹配；人工实体条件比例仅诊断误差，不作为推理输入。',['results','attributes-table'])
    query('entity_labels',[dict(label=k,**v) for k,v in ex['entities']['runs']['before']['per_label'].items()],
          [evaluation/'after_metrics.json'],'严格实体逐标签F1；测试无参考标签不可评价。',['results','entity-labels-table'])
    auxiliary=[]
    for b,m in comparison.items():
        if not b.startswith('neural_'):continue
        for kind in ('syndrome','pathogen'):
            auxiliary += [dict(model=names[b],kind=kind,**p) for p in m[kind]['per_label']]
    query('auxiliary',auxiliary,[evaluation/'comparison.json'],'辅助证据标签由人工实体属性派生；病原未知槽位不计；无阳性支持不能宣称有效。',['results','auxiliary-table'])
    params=[dict(parameter=k,value=str(v)) for k,v in er['parameters'].items()]
    query('parameters',params,[extractor/'experiment_receipt.json'],'实际选中的抽取训练参数；实体阈值只在验证集选择。',['tools','parameters-table'])
    attribute_audit=out/'pathogen_attributes.json'
    if attribute_audit.exists():
        d=read(attribute_audit);schema=read(data/'schema.json')
        field_names={'test_result':'检验结果','specimen_type':'标本类型','detection_method':'检测方式'}
        query('detection_choices',[dict(field=f,field_label=field_names[f],option_label=schema['fields'][f]['values'][c['option']],**c) for f,v in d['fields'].items() for c in v['choices']],
              [attribute_audit,data/'schema.json'],'仅人工有参考的属性槽位；实体漏检计错；各选项无参考支持保留为空。显示名称按冻结Schema解析别名，原始字段和值仍保留。',['results','detection-choices-table'])
    summary=(f"真实128例与1000条合成样本共同训练；验证、测试各27例真实病例。严格实体F1由{pct(oldmicro['f1'])}升至{pct(micro['f1'])}，变化{(micro['f1']-oldmicro['f1'])*100:+.2f}个百分点；选中抽取器的验证分数为{pct(er['best_validation_score'])}。第一轮修正模型测试78.75%，第二轮验证更高但测试降至76.44%；未按测试结果改选。\n\n"
             f"病例分类尚未达到80%—85%。验证选中的{names[selected_key]}在开发测试集macro F1为{pct(case['macro_f1'])}，原融合分类器为{pct(oldcase['macro_f1'])}。小模型对照未支持直接替换为XGBoost；本轮保留各分支结果，不按测试集最高分改选模型。所有测试结论限于已经用于开发的本批数据。")
    results=(f"实体精确率{pct(micro['precision'])}、召回率{pct(micro['recall'])}、F1 {pct(micro['f1'])}：TP {micro['tp']}、FP {micro['fp']}、FN {micro['fn']}，参考414个、32类有支持。已知关系正例召回{pct(ex['relations']['known_positive_recall'])}（{ex['relations']['known_positive_hits']}/152），真实无关系负例未全面标注，完整关系精确率/F1不可评价。\n\n"
             f"验证选中的病例模型准确率{pct(case['accuracy'])}、macro F1 {pct(case['macro_f1'])}。这没有超过原融合病例分类器的测试macro F1 {pct(oldcase['macro_f1'])}，因此本轮属于改进试验，未替换在线服务。四类真实测试支持分别为目标4、复核4、非目标13、信息不足6；单例变化会显著影响少数类指标。\n\n"
             "文本向量＋证据逻辑回归在测试表中分数较高，但它的验证分数更低；该观察只作为探索结果，不据此更换选中模型。症候群证据和病原检出头分别报告已知槽位支持，检出不能解释成肺炎病因。呼吸受损、严重全身表现及细菌/真菌没有真实阳性支持。")
    detection_note=''
    if attribute_audit.exists():
        positive=next(c for c in d['fields']['test_result']['choices'] if c['option']=='positive')
        joint=d['joint_known_record']
        detection_note=(f"病原检测三属性均已训练和输出。但阳性参考仅{positive['correct']}/{positive['support']}正确（{pct(positive['known_slot_recall'])}），总体正确比例掩盖了阳性识别不足。三项有参考的检测记录中，实体及三属性全部正确{joint['correct']}/{joint['support']}（{pct(joint['accuracy'])}）；这不是完整预测记录F1。标本参考116/117为原文未说明；mNGS、培养、双份抗体等未获真实方法支持，因此总体正确比例不能代表这些具体属性有效。")
    sections=[
        ('construction','一、建设思路回顾',['splits'], '验收分别关注实体严格micro F1与病例四分类macro F1，争取80%—85%以上。实体抽取、病例判断、症候群证据和病原检出按不同监督目标设计；分类器可独立训练并复用mmBERT预测特征。更大的分类头并不自动带来更好的真实病例效果。'),
        ('guide','二、数据集标注指南研发过程',['splits','attribute_training'], '保持49类实体、9个属性、5种关系和病例四分类的XML快照及182例人工决定不变。否定、历史、主体、条件/假设、检验阴性/未知及信息不足继续按原规范处理。增加的14个病例文本线索仅作为可学习特征，不新增实体标签，不自动重裁病例。1000条样本是按明确规则生成的训练数据，未获逐例专家签署；不能代替真实病例验收。\n\n真实训练中的检验结果为阳性20、阴性257、无法判断32；合成数据的750处病原检测属性全为痰＋核酸检测，三种结果各250。合成数量不能代表标本和方法组合的多样性；需补真实参考并通过属性损失加权与样本对照验证。'),
        ('tools','三、数据集工具与模型建设',['validation','parameters','alignment'], '本轮核查[官方mmBERT集合](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder)，公开条目为base、small及训练检查点，未找到可直接使用的官方mmBERT-large。继续使用本地mmBERT-base，不与原版ModernBERT对比。\n\n实体头增加字符边界修正、跨度宽度偏置和同类重叠抑制；训练保留可表示实体的监督，提高真实样本与难负例权重，采用较小编码器学习率、梯度累积、预热和余弦调度。实体阈值仅在验证集选择。\n\n病例头比较分层神经网络、正则化逻辑回归和浅层XGBoost，分别使用原文、mmBERT文本向量、预测结构化证据及其组合。合成样本在分类阶段比较不用和低权重使用。所有标准化、词表和特征筛选只拟合训练行。保存权重后重载，验证预测一致。病例与辅助头可使用任务独立的已冻结抽取器；这不是联合损失回传到全部编码器的端到端训练。'),
        ('annotation','四、标注情况与计划',['splits'], '182例、11845个审查点的人工确认与历史数据库已经冻结，保留全部修订，不改写人工答案。真实患者分组沿用128/27/27，训练共1128例。病例真实训练分布为非目标56、信息不足42、目标16、复核14。专家签署、独立症候群和致病原因标注仍未完成。'),
        ('results','五、初步结果',['entity_comparison','scores','case_classes','attributes','entity_labels','auxiliary']+(['detection_choices'] if attribute_audit.exists() else []),results+'\n\n'+detection_note+'后续融合尚未显式编码标本和检测方式；已查阅[Laya](https://github.com/NandhaKishorM/laya)、[PURE](https://aclanthology.org/2021.naacl-main.5/)、[SpERT](https://arxiv.org/abs/1909.07755)与[RadGraph](https://arxiv.org/abs/2106.14463)，实体条件属性查询和完整检测记录特征属于待实施设计。'),
        ('next','六、下一步计划',['scores'], '实体性能已明显改善，病例macro F1仍需更多真实监督。优先扩充目标与待专业复核病例，检查类别边界的一致性，再训练病例分类器。下一轮为病例元分类训练生成患者分组的折外抽取特征，降低训练阶段过于理想的实体输入与推理错误之间的差距；当前尚未执行折外抽取。合成模板通过低权重使用与真实单独训练对照验证，不继续单纯增加模板数量。使用未参与开发的新批次做最终验收，分别报告实体、病例、已知关系和辅助头的支持数，不把准确率或合成集分数当作80%—85%的macro F1。')]
    snapshot=dict(id='report:mmbert-optimization-20261011',surface='report',title='实体抽取已改善，病例分类仍需真实样本验证',
                  status='development_evaluation_not_external_acceptance',generatedAt=datetime.now(timezone.utc).isoformat(),buildStatus='complete' if complete else 'creating',filters=[],
                  report=dict(asOf='2026-10-11',summary=summary,sections=[dict(id=i,title=t,queryIds=q,text='## '+t+'\n\n'+x) for i,t,q,x in sections]),queries=queries)
    if args.app_dir and (args.app_dir/'src/data.json').exists():snapshot['id']=read(args.app_dir/'src/data.json')['id']
    write_json(out/'reviewed_snapshot.json',snapshot)
    if args.app_dir:write_json(args.app_dir/'src/data.json',snapshot)
    text='# '+snapshot['title']+'\n\n'+summary+'\n\n'+'\n\n'.join(s['text'] for s in snapshot['report']['sections'])
    text+='\n\n|模型|病例macro F1|准确率|\n|---|---:|---:|\n'+'\n'.join(f"|{r['model']}|{pct(r['macro_f1'])}|{pct(r['accuracy'])}|" for r in scores)+'\n'
    (out/'report.md').write_text(text,'utf-8')
    repo=Path(__file__).resolve().parents[2]
    files={f['path']:f['sha256'] for q in queries.values() for f in q['source']['files']}
    command=['python','-m','emr_annotation.evaluation.mmbert_optimization_report',
             '--batch-dir',str(root.resolve()),'--evaluation-dir',str(evaluation.resolve()),'--output-dir',str(out.resolve())]
    if args.app_dir:command+=['--app-dir',str(args.app_dir.resolve())]
    if complete:command+=['--complete']
    manifest=dict(stage='delivery',run_id=out.name,generated_at=snapshot['generatedAt'],
                  artifact_id=snapshot['id'],status=snapshot['status'],build_status=snapshot['buildStatus'],
                  command_arguments=command,input_sha256=files,
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  working_tree=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
                  code_sha256=sha256(Path(__file__)),
                  outputs={n:sha256(out/n) for n in ('report.md','reviewed_snapshot.json')},
                  validation_receipts=['pipeline_verification.json','report_render_check.json','offline_tests.log'],
                  validation_scope='See the separate current receipts; report generation does not rerun model training or clinical validation.')
    write_json(out/'report_manifest.json',manifest)
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('batch-dir','evaluation-dir','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--app-dir',type=Path);p.add_argument('--complete',action='store_true')
    a=p.parse_args();r=build(a,a.complete);print(json.dumps(dict(title=r['title'],status=r['buildStatus']),ensure_ascii=False))


if __name__=='__main__':main()
