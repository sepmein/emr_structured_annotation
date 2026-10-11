/** Create a source-backed project progress PPTX from reviewed report evidence.
 * Run from a private build copy with the bundled artifact-tool runtime.
 * No raw clinical text is read or embedded. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';

const args={};
for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['snapshot','freeze','verification','attributes','workspace','cover','skill-dir','runtime-python']){
  if(!args[key])throw new Error('Missing --'+key);
}
const workspace=path.resolve(args.workspace),build=path.join(workspace,'.codex_build');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const digest=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const snapshot=await read(args.snapshot),freeze=await read(args.freeze),verification=await read(args.verification),attributes=await read(args.attributes);
if(verification.status!=='passed'||!verification.frozen_files_unchanged)throw new Error('Unverified human reference');
const q=id=>snapshot.queries[id].rows;
const entities=q('entity_comparison'),before=entities[0],after=entities.at(-1);
const cases=q('scores'),selected=cases.find(r=>r.selected),prior=cases.find(r=>r.branch==='prior_fusion');
const semantic=cases.find(r=>r.branch==='small_semantic_fusion'),xgb=cases.find(r=>r.branch==='small_xgboost');
const positive=attributes.fields.test_result.choices.find(r=>r.option==='positive');
const points=verification.point_statuses;
if(points.accepted+points.modified!==freeze.points)throw new Error('Review point totals do not reconcile');
const utils=await import(pathToFileURL(path.join(args['skill-dir'],'container_tools/artifact_tool_utils.mjs')).href);
const font=utils.resolvePresentationFont({fontFamily:'Microsoft YaHei'});
const fontPolicy={basis:'design',families:[font]};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#143449',blue:'#176B92',teal:'#15948C',amber:'#C97C18',muted:'#647581',grid:'#DCE4E8',paper:'#FFFFFF'};
const slides=[];
const number=v=>v.toLocaleString('en-US');
const pct=v=>(v*100).toFixed(2)+'%';

function text(s,name,value,x,y,w,h,size=28,color=C.navy,bold=false,alignment='left'){
  const shape=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:font,fontSize:size,color,bold,alignment,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};
  return shape;
}
function slide(title,section,note,sources=[]){
  const s=p.slides.add();s.background.fill=C.paper;
  const index=slides.length+1;slides.push({slide:s,title,section,sources});
  text(s,'section',section,72,32,1100,30,20,C.teal,true);
  text(s,'title',title,72,82,1140,115,46,C.navy,true);
  text(s,'footer','肺炎病历结构化标注与模型建设  阶段进展',72,678,1010,23,16,C.muted);
  text(s,'page',String(index).padStart(2,'0'),1160,678,48,23,16,C.muted,false,'right');
  s.speakerNotes.text=note+'\n\n依据：\n'+sources.join('\n');
  return s;
}
function chart(s,name,categories,series,{x=72,y=202,w=770,h=382,percent=false,max=1,legend=false,type='bar',direction='column'}={}){
  // Excel's editable chart workbook supports 15 significant numeric digits.
  // Preserve the source metrics in the manifest; round only chart cell values.
  series=series.map(r=>({...r,values:r.values.map(v=>Number(v.toPrecision(15)))}));
  const ch=s.charts.add(type,{name,position:{left:x,top:y,width:w,height:h},categories,series,
    hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:22,fill:C.navy}},
    barOptions:{direction,grouping:'clustered',gapWidth:105},doughnutOptions:{holeSize:65},
    chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    xAxis:{textStyle:{fontSize:22,fill:C.navy},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{min:0,max,majorUnit:percent?.2:undefined,numberFormatCode:percent?'0%':'0',textStyle:{fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.grid,width:1}},
    dataLabels:{showValue:type!=='doughnut',showPercent:type==='doughnut',...(type==='doughnut'?{}:{position:'outEnd'}),numberFormatCode:percent?'0.00%':type==='doughnut'?'0.0%':'0',textStyle:{fontSize:24,fill:C.navy,bold:true}}});
  utils.applyPresentationChartFont(ch,{fontFamily:font});
  return ch;
}
const sources={
  report:path.resolve(args.snapshot),freeze:path.resolve(args.freeze),verification:path.resolve(args.verification),attributes:path.resolve(args.attributes),
  schema:snapshot.queries.attribute_training.source.files.find(f=>f.path.endsWith('schema.json')).path,
  entity:snapshot.queries.entity_comparison.source.files.map(f=>f.path),
  cases:snapshot.queries.scores.source.files.map(f=>f.path)
};

// 1. Minimal cover with editable foreground text and an embedded conceptual asset.
{
  const s=p.slides.add();s.background.fill='#FBFBF8';
  slides.push({slide:s,title:'肺炎病历结构化标注与模型建设进展',section:'阶段成果',sources:[sources.report]});
  const bytes=await fs.readFile(args.cover);
  s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'医疗文书与结构化信息的概念插画',fit:'cover',position:{left:0,top:0,width:1280,height:720}});
  text(s,'cover-section','阶段成果汇报',76,99,740,42,27,C.teal,true);
  text(s,'cover-title','肺炎病历结构化标注\n与模型建设进展',76,196,810,210,63,C.navy,true);
  text(s,'cover-subtitle','人工审查与训练评估流程已跑通',78,448,715,53,30,C.blue);
  text(s,'cover-status','本轮效果尚未达到80%—85%的F1目标',78,511,730,46,23,C.muted);
  text(s,'cover-date','2026年10月',78,642,600,38,24,C.navy);
  s.speakerNotes.text='汇报定位为本轮阶段进展。实体识别有实测提升，病例分类与关键属性尚待改进，80%—85%的目标没有完成。以下流程是已完成的人工审查、离线训练与评估，不包含线上部署或临床验收。\n来源：'+sources.report+'\n封面为AI生成的概念插画，非实际病历或医院场景。';
}
// 2. Clear process logic in a flat four-stage composition.
{
  const s=slide('项目建设逻辑','建设过程','项目把统一规范、人工审查、数据冻结和训练评估连接起来，形成可审查的结构化证据。结果反馈到规范与模型修订。流程已完成离线运行，线上业务接入另行验收。',[sources.report,sources.verification]);
  const stages=[['01','统一规范','明确识别对象\n保留时间与否定信息'],['02','人工审查','对照原双人标注\n确认或修改裁决'],['03','数据冻结','保存参考答案\n按患者划分数据'],['04','训练评估','比较模型结果\n查明误报与漏报']];
  stages.forEach((r,i)=>{const x=72+i*292;text(s,'stage-number-'+i,r[0],x,211,250,68,52,C.teal,true);text(s,'stage-label-'+i,r[1],x,303,270,51,35,C.navy,true);text(s,'stage-detail-'+i,r[2],x,380,258,120,27,C.muted);});
  text(s,'logic-takeaway','每个判断都能回到原文证据与人工参考',72,561,1136,63,35,C.blue,true);
}
// 3. Schema coverage and a fictional record demonstrate the benefit without jargon.
{
  const s=slide('同一套规范组织病历事实与病例判断','标注规范','成人与儿童共享当前标注Schema。四个数字分别为49类实体、9个关联属性、5类关系、4种病例判断。下方检测记录是虚构示意，仅说明结构化保留哪些信息，不展示真实病历。',[sources.schema]);
  [['49','实体标签'],['9','关联属性'],['5','关系类型'],['4','病例类别']].forEach((r,i)=>{const x=72+i*284;text(s,'schema-value-'+i,r[0],x,199,245,100,76,i===1?C.teal:C.blue,true);text(s,'schema-label-'+i,r[1],x,302,248,49,29,C.navy);});
  text(s,'fictional-note','示意文本：鼻咽拭子流感病毒核酸检测阴性',72,380,1136,47,28,C.navy);
  const table=s.tables.add({rows:2,columns:4,left:72,top:449,width:1136,height:119,columnTracks:[{mode:'fr',value:1},{mode:'fr',value:.8},{mode:'fr',value:1.5},{mode:'fr',value:1.1}],values:[['病原体','检验结果','标本类型','检测方式'],['流感病毒','阴性','口咽/鼻咽拭子','核酸检测']]});
  table.borders.assign({fill:C.grid,width:1});
  for(let r=0;r<2;r++)for(let c=0;c<4;c++){const cell=table.getCell(r,c);cell.fill=r===0?C.navy:'#F5F8FA';cell.text.style={typeface:font,fontSize:26,color:r===0?'#FFFFFF':C.navy,bold:r===0};}
  text(s,'schema-purpose','症状同时保留主体、时间与存在状态，便于核查其含义',72,600,1136,44,27,C.blue);
}
// 4. Human decisions remain visible and versioned.
{
  const s=slide('人工审查形成可回溯的训练参考','人工裁决','182例已人工确认，11845个审查点中10967个认可现有裁决、878个修改。饼图是审查处理分布，不是双人一致率或标注准确率。原标注与人工修订保留，专家签署尚未完成。',[sources.freeze,sources.verification]);
  text(s,'review-cases',number(freeze.cases),72,211,490,99,78,C.blue,true);
  text(s,'review-cases-label','例人工确认病例',74,313,500,49,29,C.navy);
  text(s,'review-points',number(freeze.points),72,394,520,92,64,C.teal,true);
  text(s,'review-points-label','个审查点逐项留存',74,488,535,49,29,C.navy);
  chart(s,'review-point-distribution',['认可现有裁决 '+number(points.accepted),'人工修改 '+number(points.modified)],[{name:'审查处理',values:[points.accepted,points.modified],points:[{idx:0,fill:C.blue},{idx:1,fill:C.amber}]}],{x:632,y:202,w:571,h:385,type:'doughnut',max:freeze.points,legend:true});
  text(s,'human-accountability','原双标与修订并存，参考答案保留冻结版本',72,608,1136,43,27,C.blue,true);
}
// 5. Real data owns validation; synthetic data supplements training.
{
  const s=slide('真实病例承担评估，合成数据补充训练','训练数据','182例真实病例按患者分组为128训练、27验证、27测试。1000条规则合成样本只进入训练，混合训练共1128例。当前测试批次已用于开发，不能当作未接触的外部验收集。',[sources.report,sources.verification]);
  chart(s,'real-case-splits',['真实训练','真实验证','真实测试'],[{name:'真实病例',values:q('splits').map(r=>r.real),fill:C.blue}],{x:72,y:202,w:717,h:350,max:160});
  text(s,'synthetic-total','1,000',875,226,340,90,69,C.teal,true);
  text(s,'synthetic-label','条规则合成病例',878,324,325,48,28,C.navy);
  text(s,'synthetic-scope','仅加入训练集\n不进入验证与测试',878,404,330,101,28,C.blue,true);
  text(s,'patient-isolation','同一患者不跨训练、验证和测试集合',72,590,1136,48,30,C.navy,true);
  text(s,'development-test-note','本批测试已参与开发，最终效果需由独立新批次验收',72,638,1136,30,20,C.muted);
}
// 6. Advantages are grounded in achieved workflow capabilities.
{
  const s=slide('项目优势体现在规范、人工控制与证据留存','建设亮点','本页凝练已实现能力，不宣称优于其他项目，也不把代码完成等同于临床验收。模型可按任务独立比较，当前试验仍使用mmBERT-base，未与原版ModernBERT对比。',[sources.report,sources.verification,sources.schema]);
  const rows=[['规范贯通','标注、转换和模型评价使用同一套字段'],['人工可控','双人原标注与人工裁决保留，修改结果可审查'],['证据可查','原文、裁决、冻结数据与模型结果可以对应'],['便于迭代','实体与病例任务分开评价，比较适合的分类模型']];
  rows.forEach((r,i)=>{const y=210+i*96;text(s,'advantage-title-'+i,r[0],72,y,295,65,35,i%2?C.teal:C.blue,true);text(s,'advantage-detail-'+i,r[1],392,y+5,815,69,28,C.navy);});
  text(s,'advantage-implication','为后续公共卫生线索复核与模型改进建立基础',72,620,1136,45,29,C.blue,true);
}
// 7. Main extraction result with every iteration shown.
{
  const delta=(after.f1-before.f1)*100;
  const s=slide('实体识别F1提高'+delta.toFixed(2)+'个百分点','关键结果','同一27例真实开发测试，参考414个实体，32类有支持。严格匹配要求实体字符起止和标签均一致。三轮分别64.09%、78.75%、76.44%。第二轮修正模型验证分数更高而选中，未因测试分数较低改回第一轮。',[...sources.entity]);
  chart(s,'entity-f1',['原抽取器','修正一轮','验证选中'],[{name:'严格实体F1',values:entities.map(r=>r.f1),fill:C.blue,points:[{idx:0,fill:'#9BAEBB'},{idx:1,fill:C.teal},{idx:2,fill:C.blue}],valuesFormatCode:'0.00%'}],{x:72,y:207,w:736,h:345,percent:true});
  text(s,'entity-delta','+'+delta.toFixed(2),892,235,320,99,67,C.blue,true);
  text(s,'entity-delta-unit','个百分点',895,340,310,45,31,C.navy);
  text(s,'entity-selected-score',pct(after.f1),892,413,320,64,45,C.teal,true);
  text(s,'entity-selected-label','验证选中模型的测试F1',895,487,307,88,26,C.navy);
  text(s,'entity-boundary','目标80%—85%仍未达标，第一轮测试78.75%仅作并列结果',72,607,1136,42,25,C.muted);
  text(s,'entity-denominator','同一27例真实病例，参考414个实体，字符范围与标签严格匹配',72,649,1136,28,19,C.muted);
}
// 8. Explain what improved, including the recall trade-off.
{
  const reduction=(before.fp-after.fp)/before.fp;
  const s=slide('实体误报减少'+(reduction*100).toFixed(2)+'%','结果解释','原抽取器与本轮验证选中抽取器在同一参考集合上比较。误报288降至113，减少175个、60.76%；漏报83增至88，增加5个。精确率53.47%升至74.26%，召回率79.95%降至78.74%。误报改善是F1提升的主要算术贡献，不是对改进机制的因果证明。',[sources.entity[0],sources.entity[1]]);
  chart(s,'entity-errors',['误报实体','漏报实体'],[{name:'原抽取器',values:[before.fp,before.fn],fill:'#9BAEBB'},{name:'验证选中',values:[after.fp,after.fn],fill:C.blue}],{x:72,y:201,w:742,h:351,max:350,legend:true});
  text(s,'precision-heading','识别结果更准确',898,232,320,53,29,C.navy,true);
  text(s,'precision-before',pct(before.precision),898,311,315,57,40,C.muted);
  text(s,'precision-after',pct(after.precision),898,399,315,76,56,C.blue,true);
  text(s,'precision-label','精确率实测变化',900,491,304,48,26,C.navy);
  text(s,'error-tradeoff','误报减少175个，漏报增加5个，召回仍需加强',72,612,1136,47,29,C.navy,true);
}
// 9. Case outcome separates exploratory scores from selection.
{
  const s=slide('病例分类尚未形成稳定提升','关键结果','同一27例真实开发测试，病例四分类macro F1。原融合67.42%，本轮验证选中59.00%，文本向量＋证据逻辑回归73.08%，XGBoost54.45%。逻辑回归测试结果更高但验证结果较低，未选为主模型。准确率81.48%不是该分支的macro F1。未用测试最高分改选模型，未替换线上服务。',[...sources.cases]);
  chart(s,'case-macro-f1',['原融合','本轮选中','文本＋证据','XGBoost'],[{name:'病例macro F1',values:[prior.macro_f1,selected.macro_f1,semantic.macro_f1,xgb.macro_f1],points:[{idx:0,fill:'#9BAEBB'},{idx:1,fill:C.blue},{idx:2,fill:C.amber},{idx:3,fill:C.teal}],valuesFormatCode:'0.00%'}],{x:72,y:214,w:1136,h:325,percent:true});
  text(s,'selection-finding','测试较高的分支保留为探索结果',72,570,1136,43,31,C.navy,true);
  text(s,'selection-boundary','模型选择依据验证集，病例分类目标仍未达到',72,624,1136,34,24,C.muted);
  text(s,'case-metric-definition','病例macro F1为四类F1平均，当前评估仅27例真实病例',72,658,1136,24,18,C.muted);
}
// 10. One limitations slide explains the most material quality gaps.
{
  const overall=attributes.fields.test_result;
  const s=slide('关键阳性与检测属性仍是明显短板','不足','检验结果端到端已知槽位正确比例107/117=91.45%；阳性参考正确1/7=14.29%，其余6条被判为阴性。两项分母不同，不是完整检测记录F1。标本参考116/117未说明，不能用97.44%总体正确比例证明各种标本有效。标本类型与检测方式尚未显式进入病例/病原融合证据。本批仅27例开发测试，未开展独立新批次验收，专家签署尚未完成。',[sources.attributes,sources.report]);
  chart(s,'detection-known-slots',['总体已知槽位','阳性参考'],[{name:'正确比例',values:[overall.accuracy,positive.known_slot_recall],points:[{idx:0,fill:C.blue},{idx:1,fill:C.amber}],valuesFormatCode:'0.00%'}],{x:72,y:222,w:650,h:335,percent:true});
  text(s,'positive-main','7条阳性仅1条识别正确',795,217,405,91,35,C.amber,true);
  text(s,'specimen-gap','标本参考116/117未说明\n具体标本能力缺少验证',795,352,405,104,27,C.navy);
  text(s,'fusion-gap','标本和检测方法尚未进入\n后续结构化判断特征',795,492,405,104,27,C.navy);
  text(s,'attribute-denominators','总体107/117，阳性1/7，均要求匹配参考实体',72,593,680,40,21,C.muted);
  text(s,'acceptance-gap','测试样本少，独立新批次验收与专家签署尚未完成',72,645,1136,31,21,C.muted);
}
// 11. One next-work slide stays at business and evidence level.
{
  const s=slide('下一阶段重点是补数据、补证据并独立验收','下一步','以下为待实施工作。实体条件属性查询与完整检测特征仅有架构分析，未实现或训练。病例元分类尚未执行患者分组折外抽取，需降低训练输入过于理想与推理错误之间的差距。最终验收需要新批次真实参考与专家校准，继续分别关注实体严格micro F1和病例macro F1达到80%—85%以上。没有承诺时限或保证分数。',[sources.report]);
  const rows=[['补真实数据','扩充目标、待复核和病原阳性病例，开展专家校准'],['补完整证据','结果、标本与方法绑定具体病原，保留不同检测记录'],['改属性判断','结合实体位置与上下文，重点改善阳性和少数属性'],['独立验收','使用未参与开发的新批次，分别评价实体与病例F1']];
  rows.forEach((r,i)=>{const y=207+i*97;text(s,'next-number-'+i,String(i+1).padStart(2,'0'),72,y,83,60,35,C.teal,true);text(s,'next-label-'+i,r[0],175,y,260,59,32,C.navy,true);text(s,'next-detail-'+i,r[1],448,y+2,756,73,27,C.navy);});
  text(s,'next-target','验收目标保留为80%—85%，按独立实测结果判断是否达标',72,615,1136,53,28,C.blue,true);
}

await fs.mkdir(build,{recursive:true});await fs.mkdir(path.join(workspace,'final'),{recursive:true});
const plan=slides.map((r,i)=>({slide:i+1,title:r.title,section:r.section,sources:r.sources}));
await fs.writeFile(path.join(workspace,'storyline.json'),JSON.stringify(plan,null,2));
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log('DRAFT_EXPORTED',slides.length,candidate);
const charts=[4,5,7,8,9,10],finalPath=path.join(workspace,'final',args['output-name']||'肺炎病历结构化项目_阶段进展汇报.pptx');
const result=await utils.finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
  pythonExecutable:args['runtime-python'],
  integrityValidatorPath:path.join(args['skill-dir'],'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(args['skill-dir'],'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3'],
  requiredNativeTableOwnerSlides:[3],requiredNativeChartOwnerSlides:charts,
  materializeLiteralChartWorkbooks:true,fontPolicy,verifyArtifactToolImport:true,
  receiptPath:path.join(build,'presentation.validation.json')});
console.log('FINALIZED',JSON.stringify(result));
// Render the finalized file, rather than assuming the in-memory draft matches it.
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(build,'previews'),{recursive:true});
for(const [i,s] of finalDeck.slides.items.entries()){
  const png=await finalDeck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(build,'previews',String(i+1).padStart(2,'0')+'.png'),new Uint8Array(await png.arrayBuffer()));
  const layout=await s.export({format:'layout'});await fs.writeFile(path.join(build,'previews',String(i+1).padStart(2,'0')+'.layout.json'),await layout.text());
  console.log('RENDERED',i+1);
}
const evidenceFiles=[args.snapshot,args.freeze,args.verification,args.attributes,args.cover];
await fs.writeFile(path.join(workspace,'presentation_manifest.json'),JSON.stringify({stage:'delivery',run_id:path.basename(workspace),status:'created_pending_visual_review',created_at:new Date().toISOString(),slides:slides.length,scope:'Project progress; target performance remains unmet',
  input_sha256:Object.fromEntries(await Promise.all(evidenceFiles.map(async f=>[path.resolve(f),await digest(f)]))),
  code_sha256:await digest(new URL(import.meta.url)),final_path:finalPath,final_sha256:await digest(finalPath),native_chart_slides:charts,native_table_slides:[3],font_policy:fontPolicy,validation_receipt:path.join(build,'presentation.validation.json'),rendered_from_final:true,powerpoint_application_check:false},null,2));
console.log('PRESENTATION_COMPLETE',finalPath);
