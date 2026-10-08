const fs=require('node:fs'),path=require('node:path');
const {chromium}=require('C:/Users/Spencer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..');
const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
(async()=>{
const b=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
const p=await b.newPage({viewport:{width:1600,height:1050},reducedMotion:'reduce'}),errors=[];
p.on('pageerror',e=>errors.push(e.message));
await p.goto('file:///'+path.join(root,'documentation/layout-preview/annotation-manual-design-b.html').replaceAll('\\','/'));
await p.locator('.navbtn[data-group="病例四分类"]').click();
assert(await p.locator('.entry').count()===4,'Expected four case cards');
for(const [id,title] of [['2.1','目标病例信号'],['2.2','待专业复核'],['2.3','非目标'],['2.4','信息不足']]){
  await p.locator('.entry[data-id="'+id+'"]').click();
  assert(await p.locator('#title').textContent()===title,'Wrong case title');
  assert((await p.locator('#meta').textContent()).includes('病例级结论'),'Wrong case-level metadata');
  assert((await p.locator('#body').textContent()).includes('外部表单'),'Evidence instructions missing');
  assert(await p.locator('#rules').count()===1&&await p.locator('#examples').count()===1&&await p.locator('#attributes').count()===1,'Case sections missing');
  await p.locator('[data-tab="examples"]').click();assert(await p.locator('#examples').isVisible(),'Case examples tab failed');
  await p.locator('[data-tab="reference"]').click();assert(await p.locator('#attributes').isVisible(),'Case reference tab failed');
  await p.locator('.navbtn[data-group="病例四分类"]').click();
}
await p.locator('.entry[data-id="2.1"]').click();await p.locator('[data-tab="operation"]').click();
await p.screenshot({path:path.join(root,'tmp/case-decision-help-desktop.png')});
await p.locator('#search').fill('目标病例信号');assert(await p.locator('.entry[data-id="2.1"]').count()===1,'Case card not searchable');
await p.locator('.navbtn[data-view="overview"]').click();assert(await p.locator('.entry').count()===49,'Entity catalogue changed');
await p.locator('.navbtn[data-group="病例四分类"]').click();await p.locator('.entry[data-id="2.4"]').click();
await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile case help overflow');
await p.goto('file:///'+path.join(root,'documentation/layout-preview/annotation-manual-preview.html').replaceAll('\\','/'));
assert(await p.locator('#case-decision-help').count()===1,'Quick preview help missing');
assert(errors.length===0,errors.join(';'));
console.log('Four case cards, definitions, tabs, search, 49-entity catalogue, mobile layout and preview help passed.');
await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
