import React from 'react';
import {DataComponent,DataTable,EvidenceChart,ReportSection,RichNarrative,useDataApp} from '../../data-app-public.jsx';

const pct=v=>v==null?'不可评价':`${(v*100).toFixed(2)}%`;
const num=v=>v==null?'—':v;
const col=(field,label,renderCell)=>({field,label,...(renderCell?{renderCell}:{}),...(renderCell===pct||renderCell===num?{align:'right'}:{})});
const tables={
 entities:['实体严格匹配：四个固定分支',[col('model','分支'),col('f1','F1',pct),col('precision','精确率',pct),col('recall','召回率',pct),col('tp','正确',num),col('fp','误报',num),col('fn','漏报',num)],'entity-table'],
 cases:['病例分类：按验证保留现有模型',[col('model','分类器'),col('validation_f1','验证F1',pct),col('oof_f1','患者3折OOF',pct),col('macro_f1','开发测试F1',pct),col('accuracy','准确率',pct),col('selected','保留',v=>v?'是':'否')],'case-table'],
 splits:['真实患者划分与合成训练数据',[col('split','集合'),col('real_cases','真实病例',num),col('synthetic_cases','合成病例',num),col('patient_groups','真实患者组',num)],'split-table'],
 causes:['错分证据与对应改进',[col('observation','直接观察或假设'),col('count','数量',num),col('denominator','比较分母',num),col('status','证据状态'),col('intervention','对应改进')],'cause-table'],
 attributes:['9属性：区分实体漏报与属性误判',[col('attribute','属性'),col('support','已知参考',num),col('before','改进前正确',num),col('after','改进后正确',num),col('upstream_errors','实体不匹配',num),col('matched_errors','实体已匹配但属性错误',num)],'attribute-table'],
 detection:['检验结果、标本与方式逐值核对',[col('field','属性'),col('option','参考选项'),col('support','已知参考',num),col('before','改进前正确',num),col('after','改进后正确',num),col('recall','逐值召回',pct)],'detection-table'],
 labels:['49类实体前后比较',[col('label','标签'),col('support','真实参考',num),col('before_f1','改进前F1',pct),col('after_f1','改进后F1',pct),col('tp','正确',num),col('fp','误报',num),col('fn','漏报',num)],'label-table']
};
const previews={
 'https://aclanthology.org/P19-1091/':{title:'临床实体抽取与断言检测联合模型',summary:'研究共享编码器与条件解码的联合任务。本轮注意力属性头未取得验证改善，论文方向不能替代实测。',approvedForReport:true},
 'https://arxiv.org/abs/2208.03054':{title:'GlobalPointer跨度模型',summary:'提供跨度建模和不平衡损失思路；本轮没有实施其原损失。',approvedForReport:true},
 'https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder':{title:'官方mmBERT集合',summary:'本次核查可见small/base，尚未确认官方large可用权重。',date:'2026-10-12',approvedForReport:true}
};

export function ReportContent(){
 const {snapshot,reviewedPeriodRows,visible,canEdit,mode,appTitle,setAppTitle}=useDataApp();
 const rows=k=>reviewedPeriodRows(k);
 const table=k=>{const [title,columns,id]=tables[k];return visible(id)&&<DataComponent id={id} queryId={k} kind="table" title={title} sourceRows={rows(k)} displayRows={rows(k)}><DataTable rows={rows(k)} columns={columns} label={title} searchable={['detection','labels'].includes(k)}/></DataComponent>};
 return <article className="report-content error-driven-report">
  <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==='edit'} suppressContentEditableWarning onBlur={canEdit&&mode==='edit'?e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1></header>
  {visible('report-summary')&&<ReportSection id="report-summary" title="本轮结论" queryId="entities" queryIds={['entities','cases','splits']} sourceRowsByQuery={{entities:rows('entities'),cases:rows('cases'),splits:rows('splits')}} showHeading={false}><RichNarrative id="report-summary:body" value={'## 本轮结论\n\n'+snapshot.report.summary}/></ReportSection>}
  {snapshot.report.sections.map(s=><React.Fragment key={s.id}>
   {visible(s.id)&&<ReportSection id={s.id} title={s.title} queryId={s.queryIds[0]} queryIds={s.queryIds} sourceRowsByQuery={Object.fromEntries(s.queryIds.map(k=>[k,rows(k)]))} showHeading={false}><RichNarrative id={s.id+':body'} value={s.text} sourcePreviews={previews}/></ReportSection>}
   {s.id==='construction'&&table('causes')}
   {s.id==='annotation'&&table('splits')}
   {s.id==='results'&&<>
    {visible('entity-chart')&&<EvidenceChart id="entity-chart" queryId="entities" title="实体严格F1：组合方案达到82.55%" spec={{type:'bar',x:'model_short',y:'EntityF1Rate',stackable:false,valueDecimals:2}} rows={rows('entities').map(r=>({...r,EntityF1Rate:r.f1}))} sourceRows={rows('entities')} height={340}/>}
    {table('entities')}
    {visible('case-chart')&&<EvidenceChart id="case-chart" queryId="cases" title="病例分类：新试验仍低于80%" spec={{type:'horizontalBar',x:'model',y:'CaseF1Rate',stackable:false,valueDecimals:2}} rows={rows('cases').map(r=>({...r,CaseF1Rate:r.macro_f1}))} sourceRows={rows('cases')} height={340}/>}
    {table('cases')}{table('attributes')}{table('detection')}{table('labels')}
    <RichNarrative id="error-review-links" value={'### 逐项审查入口\n\n[改进前完整错分清单](../../before_review.html) · [改进后完整错分清单](../../after_review.html) · [参考疑点清单](../../reference_questions.json)\n\n清单支持病例、实体、属性和关键词筛选，以及上下翻页。它用于只读模型错分审查；人工裁决工具及已保存结果保持原样。'}/>
   </>}
  </React.Fragment>)}
 </article>;
}
