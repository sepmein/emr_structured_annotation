// Bounded local report/ledger rendering check. Never reads real records into source.
const {chromium}=require('C:/Users/Spencer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
const root=process.argv[2],url=process.argv[3];
if(!root||!/^http:\/\/127\.0\.0\.1:\d+\//.test(url))throw Error('Supply delivery directory and local HTTP report URL');
(async()=>{
 const source=JSON.parse(fs.readFileSync(path.join(root,'report_app/src/data.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Users/Spencer/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(url,{waitUntil:'networkidle'});await page.locator('.error-driven-report').waitFor();
 if(await page.locator('h1[data-data-app-title]').innerText()!==source.title)throw Error('Rendered title mismatch');
 if(source.buildStatus!=='complete')throw Error('App still creating or updating');
 const checks={};
 for(const [id,key,field,direction] of [['entity-chart','entities','f1','height'],['case-chart','cases','macro_f1','width']]){
  const card=page.locator(`[data-component-id="${id}"]`);await card.scrollIntoViewIfNeeded();
  const expected=source.queries[key].rows.map(r=>r[field]);
  const sizes=await card.locator('svg').evaluate((svg,dimension)=>[...svg.querySelectorAll('path.recharts-rectangle')].map(r=>Number(r.getAttribute(dimension))),direction);
  if(sizes.length!==expected.length||sizes.some(v=>v<=0))throw Error('Missing visible chart marks: '+id);
  const maxSize=Math.max(...sizes),maxValue=Math.max(...expected);
  if(sizes.some((v,i)=>Math.abs(v/maxSize-expected[i]/maxValue)>1e-3))throw Error('Chart scale does not match reviewed values: '+id);
  checks[id]={visibleMarks:sizes.length,values:expected};await card.screenshot({path:path.join(root,id+'.png')});
 }
 const chart=page.locator('[data-component-id="entity-chart"]');await chart.scrollIntoViewIfNeeded();
 await chart.getByRole('button',{name:'实体严格F1：组合方案达到82.55% actions',exact:true}).click();
 await page.getByRole('menuitem',{name:'View data source',exact:true}).click();
 await page.waitForFunction(()=>document.body.innerText.includes('2TP/(2TP+FP+FN)'));
 checks.sourceInspection=true;await page.locator('button[aria-label="Close data source"]:visible').last().click();
 await page.locator('.source-sidebar-layer').waitFor({state:'hidden'});
 const detection=page.locator('[data-component-id="detection-table"]');await detection.scrollIntoViewIfNeeded();
 await detection.getByPlaceholder('Search data').fill('阳性');
 if(await detection.locator('tbody tr').count()!==1||!(await detection.innerText()).includes('100.00%'))throw Error('Positive-value search mismatch');
 await detection.getByPlaceholder('Search data').fill('');checks.detectionSearchReset=true;
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(root,'report_desktop.png')});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Desktop overflow');
 await page.setViewportSize({width:390,height:844});await page.reload({waitUntil:'networkidle'});await page.locator('.error-driven-report').waitFor();
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
 for(const id of ['entity-chart','case-chart']){const card=page.locator(`[data-component-id="${id}"]`);await card.scrollIntoViewIfNeeded();if(await card.locator('path.recharts-rectangle').count()!==4)throw Error('Mobile marks missing');await card.screenshot({path:path.join(root,id+'_mobile.png')});}
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(root,'report_mobile.png')});
 const base=new URL('../../',url).href;
 for(const prefix of ['before','after']){
  await page.goto(base+prefix+'_review.html',{waitUntil:'networkidle'});
  const ledger=JSON.parse(fs.readFileSync(path.join(root,prefix+'_errors.json'),'utf8'));
  const initial=await page.locator('article').first().getAttribute('id');
  await page.locator('header .next').click();
  if(await page.locator('article').first().getAttribute('id')===initial)throw Error('Ledger pagination did not advance');
  await page.locator('#domain').selectOption('case');
  if(await page.locator('article').count()!==ledger.filter(r=>r.domain==='case').length)throw Error('Case filter incomplete');
  await page.locator('#domain').selectOption('');
  if(await page.locator('article').count()!==Math.min(50,ledger.length))throw Error('All filter reset incomplete');
  await page.locator('#search').fill('no_such_case_or_record');
  if(await page.locator('article').count()!==0)throw Error('Empty search not empty');
  await page.locator('#search').fill('');
  const pages=Math.ceil(ledger.length/50);let reviewed=await page.locator('article').count();
  for(let i=1;i<pages;i++){await page.locator('header .next').click();reviewed+=await page.locator('article').count();}
  if(reviewed!==ledger.length)throw Error('Some records inaccessible via pagination');
  checks[prefix+'Ledger']={records:ledger.length,allRecordsAccessible:true,domainAndSearchReset:true};
 }
 if(errors.length)throw Error('Browser runtime errors: '+errors.join(';'));
 fs.writeFileSync(path.join(root,'report_render_check.json'),JSON.stringify({status:'passed',url,artifactId:source.id,charts:checks,desktopWidth:1440,mobileWidth:390,consoleErrors:errors,notPerformed:['External acceptance','Online Label Studio replacement','Publication']},null,2));
 console.log(JSON.stringify(checks));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
