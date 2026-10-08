/* Rebuild the standalone guide with Node.js and marked; no browser network requests. */
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
let marked;
try { ({marked} = require('marked')); }
catch { ({marked} = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked'))); }
const root = path.resolve(__dirname, '../../..');
const sourcePath = path.join(root, 'documentation/annotation/annotation_guide_v2.1.1.md');
const outputPath = path.join(root, 'documentation/annotation/guide.html');
const source = fs.readFileSync(sourcePath, 'utf8').replace(/\r\n/g, '\n');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const plain = s => s.replace(/<br\s*\/?>/g, '\n').replace(/[`*_]/g, '');
const title = source.match(/^# (.+)/m)[1];
const chunks = source.split(/(?=^## )/m);
const sections = [], headings = [], links = [], diagrams = [];
let diagramIndex = 0, currentSection;

function sectionId(text) {
  const number = text.match(/^(\d+)\./);
  if (number) return `chapter-${number[1]}`;
  if (/^附录A/.test(text)) return 'appendix-a';
  if (/^附录B/.test(text)) return 'appendix-b';
  if (/^附录C/.test(text)) return 'appendix-c';
  return {'开始标注前，请先记住':'before-start','文件使用说明':'document-notes','目录':'contents','参考框架与外部项目':'references'}[text];
}
for (const chunk of chunks.slice(1)) {
  const text = chunk.match(/^## (.+)/)[1];
  const id = sectionId(text);
  if (!id) throw new Error(`Unmapped section: ${text}`);
  sections.push({id, title:text, source:chunk, headings:[]});
}

function parseGraph(code) {
  const nodes = new Map(), edges = [];
  const node = '([A-Za-z][A-Za-z0-9]*)(?:\\["(.*?)"\\]|\\{"(.*?)"\\})?';
  const pattern = new RegExp(`^\\s*${node}\\s*-->(?:\\|([^|]+)\\|)?\\s*${node}\\s*$`);
  for (const line of code.trim().split('\n').slice(1)) {
    const m = line.match(pattern);
    if (!m) throw new Error(`Unsupported diagram syntax: ${line}`);
    for (const offset of [1, 5]) {
      const id = m[offset], text = m[offset + 1] ?? m[offset + 2];
      if (text !== undefined) nodes.set(id, {id, text, decision:m[offset + 2] !== undefined});
    }
    edges.push({from:m[1], to:m[5], label:m[4] || ''});
  }
  return {nodes, edges};
}
// Coordinates affect presentation only. Nodes, labels and arrows come from source Mermaid.
const layouts = [
  {width:970, height:1110, nodes:{
    A:[260,65,310,76], P:[750,65,340,100], B:[260,215,310,76], I:[750,215,250,76],
    C:[260,405,440,126], C1:[750,405,320,90], N:[750,585,250,76],
    D:[260,715,440,112], T:[750,715,250,76], R:[260,950,440,112], V:[750,950,250,76]},
    routes:{'C1:N':[[910,405],[945,405],[945,585],[875,585]],
      'C1:D':[[750,450],[750,505],[530,505],[530,635],[260,635],[260,659]],
      'R:N':[[40,950],[20,950],[20,585],[625,585]]},
    labels:{'C1:N':[918,520], 'C1:D':[640,490], 'R:N':[45,915]}},
  {width:800, height:570, nodes:{F:[145,60,200,65], T:[630,60,240,65], D:[145,165,200,65], W:[630,165,240,65],
    A:[145,270,200,65], H:[630,270,240,65], X:[145,375,200,65], Y:[630,375,240,65],
    S:[145,490,200,65], V:[630,465,240,65], U:[630,540,240,45]},
    routes:{'S:V':[[245,490],[380,490],[380,465],[510,465]],'S:U':[[245,490],[330,490],[330,540],[510,540]]},
    labels:{'S:V':[435,450],'S:U':[430,525]}},
  {width:760, height:910, nodes:Object.fromEntries('ABCDEFGH'.split('').map((id,i)=>[id,[380,55+i*112,620,76]]))},
  {width:1010, height:1250, nodes:{A:[350,65,430,76], B:[350,180,430,76], C:[350,295,430,76],
    D:[350,420,350,86], E:[805,420,340,90], F:[350,585,500,108], G:[350,735,430,76],
    H:[350,865,390,86], I:[805,865,350,116], J:[350,1030,460,76], K:[350,1180,600,100]},
    routes:{'E:C':[[975,420],[995,420],[995,295],[565,295]], 'I:F':[[980,865],[995,865],[995,585],[600,585]]}}
];
function wrappedText(text, width) {
  const maxUnits = (width - 36) / 18;
  return plain(text).split('\n').flatMap(line => {
    const lines = []; let row = '', units = 0;
    for (const c of line) {
      const size = /[\x00-\xff]/.test(c) ? 0.53 : 1;
      if (units + size > maxUnits && row) { lines.push(row); row = ''; units = 0; }
      row += c; units += size;
    }
    lines.push(row); return lines;
  });
}
function diagram(code) {
  const index = diagramIndex++, graph = parseGraph(code), layout = layouts[index];
  if (!layout || graph.nodes.size !== Object.keys(layout.nodes).length || [...graph.nodes.keys()].some(id=>!layout.nodes[id]))
    throw new Error('Diagram structure changed; update its presentation layout.');
  diagrams.push({index:index+1, nodes:graph.nodes.size, edges:graph.edges.length, source:code});
  const marker = `arrow-${index}`;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${layout.width} ${layout.height}" role="img" aria-labelledby="diagram-title-${index} diagram-desc-${index}"><title id="diagram-title-${index}">图${index+1}：${['四分类判断流程','关系方向示例','一份记录从打开到提交','从培训到正式标注'][index]}</title><desc id="diagram-desc-${index}">${esc(graph.edges.map(e=>`${plain(graph.nodes.get(e.from).text)}，${e.label ? e.label+'：' : '到'}${plain(graph.nodes.get(e.to).text)}`).join('；'))}</desc><defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#718d94"/></marker></defs>`;
  for (const e of graph.edges) {
    const a = layout.nodes[e.from], b = layout.nodes[e.to], key = `${e.from}:${e.to}`;
    let points = layout.routes?.[key];
    if (!points) {
      points = a[0] === b[0] ? [[a[0], a[1]+a[3]/2], [b[0], b[1]-b[3]/2]] : [[a[0]+a[2]/2,a[1]], [b[0]-b[2]/2,b[1]]];
    }
    const d = points.map((p,i)=>`${i?'L':'M'} ${p.join(' ')}`).join(' ');
    const mid = layout.labels?.[key] || [(points[0][0]+points.at(-1)[0])/2+13,(points[0][1]+points.at(-1)[1])/2-10];
    svg += `<path d="${d}" fill="none" stroke="white" stroke-width="8"/><path class="flow-edge" data-from="${e.from}" data-to="${e.to}" d="${d}" fill="none" stroke="#718d94" stroke-width="2" marker-end="url(#${marker})"/>`;
    if (e.label) svg += `<text x="${mid[0]}" y="${mid[1]}" text-anchor="middle" class="edge-label">${esc(e.label)}</text>`;
  }
  for (const n of graph.nodes.values()) {
    const [x,y,w,h] = layout.nodes[n.id], text = wrappedText(n.text,w), special = ['P','N','I','T','V'].includes(n.id) && index===0;
    const color = n.id === 'P' ? '#fff3e8' : '#e9f3ee';
    svg += `<g class="flow-node" data-node="${n.id}"><rect x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" rx="${n.decision?22:8}" fill="${special?color:n.decision?'#f2f6f7':'#fff'}" stroke="${special?'#89ae9b':'#b3c6cb'}" stroke-width="1.6"/>`;
    if(n.decision) svg += `<path d="M ${x-w/2+13} ${y-5} l 5 5 l -5 5 l -5 -5 z" fill="#6b8f91"/>`;
    svg += `<text text-anchor="middle" x="${x}" y="${y-(text.length-1)*13+6}" fill="#28444e" font-size="18">${text.map((line,i)=>`<tspan x="${x}" dy="${i?26:0}">${esc(line)}</tspan>`).join('')}</text></g>`;
  }
  return `<figure class="flowchart"><div class="diagram-scroll" tabindex="0" aria-label="流程图，可横向滚动">${svg}</div><figcaption>图${index+1} · 按原稿节点和箭头绘制；窄屏可横向滚动。</figcaption><details class="diagram-source"><summary>查看原始流程图文本</summary><pre><code>${esc(code)}</code></pre></details></figure>`;
}

marked.use({gfm:true, renderer:{
  heading(token) {
    const text = plain(token.text), num = text.match(/^(\d+(?:\.\d+)+)/), caseNo = text.match(/^案例(\d+)：/);
    const id = token.depth===1 ? 'page-title' : token.depth===2 ? currentSection.id : caseNo ? `case-${caseNo[1]}` : num ? `s-${num[1].replaceAll('.','-')}` : `${currentSection.id}-h-${currentSection.headings.length+1}`;
    const h = {id, text, depth:token.depth, section:currentSection?.id};
    headings.push(h); if(token.depth>2) currentSection.headings.push(h);
    return `<h${token.depth} id="${id}">${this.parser.parseInline(token.tokens)}${token.depth>1?`<a class="anchor" href="#${id}" aria-label="链接到${esc(text)}">#</a>`:''}</h${token.depth}>\n`;
  },
  code(token) {
    return token.lang==='mermaid' ? diagram(token.text) : `<pre><code>${esc(token.text)}</code></pre>`;
  },
  table(token) {
    const header = token.header.map(c=>`<th scope="col">${this.parser.parseInline(c.tokens)}</th>`).join('');
    const rows = token.rows.map(row=>`<tr>${row.map(c=>`<td>${this.parser.parseInline(c.tokens)}</td>`).join('')}</tr>`).join('');
    return `<div class="table-scroll" tabindex="0" aria-label="表格，可横向滚动"><table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>\n`;
  },
  blockquote(token) {
    let html = this.parser.parse(token.tokens), type = token.text.match(/^\[!(\w+)\]/)?.[1];
    if (type) {
      html = html.replace(/<p>\[!\w+\]\s*/, '<p>');
      return `<aside class="callout ${type.toLowerCase()}" role="note"><div class="callout-label">${{IMPORTANT:'操作重点',CAUTION:'隐私与安全',WARNING:'提交前核对',NOTE:'阅读提示',TIP:'操作提示'}[type]||esc(type)}</div>${html}</aside>`;
    }
    return `<blockquote>${html}</blockquote>`;
  },
  link(token) {
    let href = token.href;
    const content = this.parser.parseInline(token.tokens);
    if (href.startsWith('/Users/sepmein/x/projects/emr_structured_annotation/')) {
      const repoPath = href.split('emr_structured_annotation/')[1];
      if (!fs.existsSync(path.join(root,repoPath))) return `<span class="unavailable" title="源稿引用的历史文件当前不在仓库中">${content}<small>（历史文件未找到）</small></span>`;
      href = '../../'+repoPath;
    }
    links.push(href);
    return `<a href="${esc(href)}"${token.title?` title="${esc(token.title)}"`:''}${/^https?:/.test(href)?' target="_blank" rel="noopener noreferrer"':''}>${content}</a>`;
  }
}});
const hero = marked.parse(chunks[0]);
for (const s of sections) {
  currentSection = s;
  s.html = marked.parse(s.source);
}
const numbered = sections.filter(s=>/^chapter-/.test(s.id));
const cases = headings.filter(h=>/^case-/.test(h.id));
const nav = ids => sections.filter(s=>ids.includes(s.id)||(ids.includes('appendix-a')&&s.id.startsWith('appendix-'))).map(s=>`<a class="chapter-link" href="#${s.id}">${esc(s.title)}</a>`).join('');
const directory = sections.filter(s=>/^chapter-|^appendix-|^references$/.test(s.id)).map(s=>`<a href="#${s.id}"><span>${esc(s.title)}</span><b aria-hidden="true">↗</b></a>`).join('');
const caseIndex = `<details class="case-index"><summary>按案例查阅 <span>28 个完整案例</span></summary><div>${cases.map(h=>`<a href="#${h.id}">${esc(h.text)}</a>`).join('')}</div></details>`;
const body = sections.map(s=>`<section class="chapter" data-section="${s.id}" aria-labelledby="${s.id}">${s.id==='contents'?`<h2 id="contents">目录<a class="anchor" href="#contents" aria-label="链接到目录">#</a></h2><div class="directory">${directory}</div>`:s.html.replace(/(<h2[^>]*>.*?<\/h2>)/s,`$1${s.id==='chapter-12'?caseIndex:''}`)}</section>`).join('\n');
const css = `
:root{--nav:#152d3a;--ink:#263e4b;--muted:#677d88;--line:#dfe7e9;--green:#157665;--orange:#b56d41;--back:#f2f5f5;--font:16px}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:100px}body{margin:0;color:var(--ink);background:var(--back);font:var(--font)/1.95 "Microsoft YaHei","PingFang SC",sans-serif}a{color:var(--green);text-underline-offset:3px}button,input{font:inherit}button{cursor:pointer}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #ce986c;outline-offset:3px}button:disabled{opacity:.45;cursor:default}.skip{position:fixed;top:-80px;z-index:30;background:white;padding:10px}.skip:focus{top:5px}.shell{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}.sidebar{height:100vh;overflow:auto;position:sticky;top:0;background:var(--nav);color:white;padding:30px 20px 20px;scrollbar-width:thin;scrollbar-color:#47606c transparent}.brand{display:flex;align-items:center;gap:12px;text-decoration:none;color:white;line-height:1.5;margin-bottom:22px}.brand .symbol{width:44px;height:44px;display:grid;place-items:center;background:#284955;border-radius:10px;font-size:29px}.brand strong{color:inherit;display:block;font-size:16px;font-weight:600}.brand small{font-size:10px;letter-spacing:1.3px;color:#a2bac5}.nav-label{color:#8faab7;font-size:11px;letter-spacing:2px;margin:23px 11px 8px}.chapter-link{display:block;border-left:3px solid transparent;text-decoration:none;color:#c0d0d8;padding:8px 11px;line-height:1.6;font-size:13px;border-radius:4px;margin:3px 0}.chapter-link:hover{background:#23414e;color:white}.chapter-link.active{background:#2d4c59;color:white;border-color:#d3a079}.side-related{margin-top:26px;padding:20px 10px 8px;border-top:1px solid #35505d;color:#9eb4bf;font-size:12px}.side-related a{color:#d7e8e0;display:block;margin:5px 0}.workspace{min-width:0}.topbar{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);min-height:74px;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:12px 34px}.breadcrumbs{font-size:12px;color:var(--muted);white-space:nowrap}.tools{display:flex;gap:9px;align-items:center}.tool{border:1px solid var(--line);border-radius:5px;background:white;color:#46606d;padding:6px 11px;white-space:nowrap;font-size:13px;text-decoration:none}.tool:hover{background:#f1f6f4}.searchbox{display:flex;gap:8px;align-items:center}.searchbox input{border:1px solid #d8e2e5;background:#f4f7f7;border-radius:5px;padding:7px 12px;width:228px;font-size:13px}.search-status{font-size:12px;white-space:nowrap;color:var(--muted)}.match-tools{display:flex;gap:5px}.match-tools button{padding:3px 9px}.progress{position:absolute;left:0;bottom:-1px;height:2px;background:#3e907a;width:0}.layout{display:grid;grid-template-columns:minmax(0,900px) 195px;gap:32px;max-width:1290px;margin:auto;padding:42px 34px 50px}.reading{min-width:0}.eyebrow{display:flex;align-items:center;gap:12px;color:var(--orange);font-size:12px;letter-spacing:2px}.eyebrow:before{content:'';width:26px;height:2px;background:var(--orange)}.hero{padding:0 5px 20px}h1{font-size:36px;font-weight:600;line-height:1.45;letter-spacing:-.5px;margin:14px 0 23px;max-width:760px}.hero>p{font-size:13px;line-height:2.05;color:var(--muted);margin:0}.stats{display:flex;gap:28px;border-top:1px solid #d5e1e1;border-bottom:1px solid #d5e1e1;padding:15px 0;margin:25px 0 21px}.stats div{display:flex;align-items:baseline;gap:8px;font-size:12px;color:var(--muted)}.stats strong{font:29px/1.4 Georgia,serif;color:#407a69}.quick-start{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:0 0 21px}.quick-start a{padding:15px;background:white;border:1px solid var(--line);border-top:2px solid #6b9b8d;border-radius:5px;text-decoration:none;font-size:13px;line-height:1.7}.quick-start small{display:block;color:var(--muted);font-size:11px;margin-bottom:4px}.quick-start a:hover{background:#edf6f0}.version-note{border-left:3px solid #c99c71;background:#faf3e8;padding:15px 18px;color:#805d40;font-size:13px;line-height:1.85;margin:20px 0 28px}.version-note strong{display:block;margin-bottom:4px;font-weight:600}.chapter{background:white;border:1px solid var(--line);border-radius:8px;padding:29px 32px;margin:0 0 22px}.chapter>h2:first-child{margin-top:0;padding-top:0;border:0}h2{font-size:23px;font-weight:600;line-height:1.6;margin:0 0 22px;letter-spacing:-.2px}h3{font-size:18px;font-weight:600;margin:33px 0 15px;padding-top:14px;border-top:1px solid #e8eeee;line-height:1.7}h4{font-size:16px;margin:25px 0 12px;font-weight:600;line-height:1.7}h2,h3,h4{scroll-margin-top:12px}.anchor{opacity:0;color:#9ab4ac;text-decoration:none;font-weight:400;padding-left:9px;font-size:15px}h2:hover .anchor,h3:hover .anchor,h4:hover .anchor,.anchor:focus-visible{opacity:1}p{margin:12px 0 16px}.reading a{overflow-wrap:anywhere}li{margin:7px 0;padding-left:3px}ul,ol{padding-left:24px;margin:13px 0 20px}li>ul,li>ol{margin:5px 0}strong{font-weight:600;color:#243f4b}code{font-family:Consolas,"Microsoft YaHei",monospace;font-size:.87em;background:#eff4f2;padding:2px 5px;border-radius:3px;color:#276d5a;overflow-wrap:anywhere}pre{background:#f3f6f6;border:1px solid #e1e9e9;padding:15px;overflow:auto;font-size:12px;line-height:1.8}pre code{background:none;padding:0;white-space:pre;overflow-wrap:normal}.table-scroll{overflow-x:auto;margin:20px 0 25px;scrollbar-width:thin;border:1px solid #dce6e6;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.85;min-width:560px}th,td{text-align:left;vertical-align:top;padding:13px 14px;border-bottom:1px solid #e2e9e9;overflow-wrap:anywhere}th{color:#4e7065;background:#eef4f1;font-weight:600}tr:last-child td{border:0}tbody tr:nth-child(even){background:#fafcfc}td:first-child{font-weight:500}blockquote{margin:18px 0;padding:16px 20px;background:#f2f7f5;border-left:3px solid #7ba28f;color:#426657}blockquote p{margin:0}.callout{background:#f1f7f4;border-left:3px solid #6d9d86;border-radius:3px;margin:18px 0 22px;padding:17px 20px;font-size:14px}.callout p:first-of-type{margin-top:3px}.callout p:last-child,.callout ol:last-child,.callout ul:last-child{margin-bottom:0}.callout-label{font-size:11px;letter-spacing:1.5px;color:#3c7a63;font-weight:600;margin-bottom:5px}.callout.caution,.callout.warning{background:#fbf2e9;border-color:#c69972}.callout.caution .callout-label,.callout.warning .callout-label{color:#93603f}.callout.note{background:#f0f5f8;border-color:#91adbb}.callout.note .callout-label{color:#526f80}hr{border:0;border-top:1px solid #e4eceb;margin:30px 0}.chapter>hr:last-child{display:none}.directory{display:grid;grid-template-columns:1fr 1fr;gap:0 24px}.directory a{display:flex;justify-content:space-between;gap:12px;text-decoration:none;border-bottom:1px solid #e7eeee;padding:12px 0;font-size:14px}.directory a:hover{color:#b77749}.directory b{font-weight:400;color:#8caaa0}.case-index{border:1px solid #cbdcd4;border-radius:5px;background:#f7faf8;margin:0 0 25px}.case-index summary{cursor:pointer;padding:13px 16px;color:#35745e;font-size:14px}.case-index summary span{float:right;color:#7a9087;font-size:12px}.case-index>div{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px;padding:6px 17px 18px}.case-index a{font-size:13px;text-decoration:none;line-height:1.7;padding:4px 0}.case-index a:hover{text-decoration:underline}#chapter-12~h4{color:#276c56}h4[id^="case-"]{padding:14px 16px;border:1px solid #dce8e1;background:#f1f6f3;border-radius:5px;margin-top:32px}h4[id^="case-"]+blockquote{border:1px solid #e1e9e6;border-left:3px solid #70a48b;background:#fbfdfb}.unavailable{color:#778993}.unavailable small{font-size:10px;display:block;color:#92724f}.toc{align-self:start;position:sticky;top:112px;max-height:calc(100vh - 140px);overflow:auto;scrollbar-width:thin;font-size:12px;color:var(--muted);padding-top:5px}.toc-label{font-size:11px;letter-spacing:2px;margin-bottom:13px}.toc-chapter{font-size:13px;color:#3e6260;padding-bottom:13px;border-bottom:1px solid #dbe5e3;margin-bottom:10px}.toc a{display:block;color:var(--muted);text-decoration:none;border-left:1px solid #d0dddc;padding:7px 12px;line-height:1.7}.toc a.active{color:var(--green);border-left:2px solid var(--green);background:#eaf1ee}.toc a.sub{font-size:11px;padding-left:20px}.toc a:hover{color:var(--green)}.toc-note{font-size:11px;margin-top:23px;line-height:1.85;border-top:1px solid #dbe5e3;padding-top:17px}.flowchart{margin:24px 0;border:1px solid #dfe9e7;border-radius:6px;background:#fcfefd;padding:14px}.diagram-scroll{overflow-x:auto;background:white;border-radius:4px}.diagram-scroll svg{display:block;width:100%;min-width:660px;max-height:none;font-family:"Microsoft YaHei",sans-serif}.edge-label{font-size:17px;fill:#637d83;paint-order:stroke;stroke:white;stroke-width:7;stroke-linejoin:round}.flowchart figcaption{font-size:11px;color:#7d938e;padding:9px 3px 0}.diagram-source{font-size:11px;color:var(--muted);margin-top:6px}.diagram-source summary{cursor:pointer}.diagram-source pre{font-size:11px}footer{padding:0 38px 25px;color:#7b8e95;font-size:11px}.back-top{position:fixed;right:22px;bottom:22px;width:38px;height:38px;background:#e8f1ee;border:1px solid #c4d8d0;border-radius:7px;color:#367561;z-index:8}.mobile-menu,.veil{display:none}mark.search-hit{background:#ffedb2;color:inherit;border-radius:2px;padding:1px 0}mark.search-hit.current{background:#edb653;outline:2px solid #b77b24}.search-live{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}[hidden]{display:none!important}
.version-note.unified{background:#edf5f1;border-color:#73a08a;color:#496e5f}
@media(min-width:1700px){.layout{max-width:1390px;grid-template-columns:minmax(0,1000px) 205px;gap:40px}.chapter{padding:36px 44px}}
@media(max-width:1280px){.layout{grid-template-columns:minmax(0,1fr);padding:32px 28px}.toc{display:none}.shell{grid-template-columns:230px minmax(0,1fr)}.topbar{padding:12px 24px}.breadcrumbs{display:none}.chapter{padding:26px}h1{font-size:32px}.tools{width:100%;justify-content:flex-end}}
@media(max-width:760px){:root{--font:15px}.shell{display:block}.sidebar{position:fixed;z-index:25;left:0;top:0;bottom:0;width:min(290px,85vw);transform:translateX(-100%);transition:transform .18s}.nav-open .sidebar{transform:translateX(0)}.nav-open .veil{display:block;position:fixed;inset:0;background:#162c3b88;border:0;z-index:24}.topbar{padding:10px 14px;gap:8px;flex-wrap:wrap;min-height:65px}.mobile-menu{display:block}.tools{width:auto;gap:7px;flex:1;justify-content:flex-end;flex-wrap:wrap}.searchbox{flex:1;min-width:135px}.searchbox input{width:100%;min-width:80px}.text-size{display:none}.source-button{display:none}.search-status{font-size:10px}.layout{padding:26px 14px}.hero{padding:0}h1{font-size:28px;line-height:1.5}.hero>p{font-size:11px}.stats{gap:24px;margin:20px 0}.stats div{display:block}.stats strong{display:block}.quick-start{gap:7px}.quick-start a{padding:11px;font-size:12px}.quick-start small{font-size:10px}.chapter{padding:22px 18px;margin-bottom:16px}h2{font-size:21px}h3{font-size:17px}.directory,.case-index>div{grid-template-columns:1fr}.case-index summary span{font-size:10px}.table-scroll{margin-right:-4px}.flowchart{padding:7px}.diagram-scroll svg{min-width:650px}.version-note{padding:13px 14px}footer{padding:0 20px 25px}.back-top{right:12px;bottom:12px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.sidebar{transition:none}}
@media print{.skip{display:none!important}}
@media print{@page{size:A4;margin:17mm 15mm}.sidebar,.topbar,.toc,.quick-start,.back-top,.veil,.anchor,.diagram-source,footer{display:none!important}body{background:white;font-size:10pt;line-height:1.65;-webkit-print-color-adjust:exact;print-color-adjust:exact}.shell,.layout{display:block;padding:0;max-width:none}.hero{padding:0}.hero h1{font-size:22pt}.hero>p{font-size:9pt}.stats{padding:8px 0;margin:15px 0}.version-note{font-size:9pt;margin:12px 0}.chapter{border:0;border-radius:0;padding:0;margin:20px 0;break-before:auto}h2{font-size:16pt;border-top:1px solid #b8ccc3!important;padding-top:16px!important;margin-top:25px!important}h3{font-size:12pt}h4{font-size:11pt}h2,h3,h4{break-after:avoid;scroll-margin-top:0}p,li{orphans:3;widows:3}blockquote,.callout,tr{break-inside:avoid}.table-scroll{overflow:visible;border:0}table{min-width:0;font-size:8.5pt;table-layout:fixed}td,th{padding:7px;overflow-wrap:anywhere}thead{display:table-header-group}.flowchart{break-inside:avoid;padding:4px;margin:15px 0}.diagram-scroll{overflow:visible}.diagram-scroll svg{min-width:0;width:100%;max-height:230mm}.case-index{display:none}.directory{grid-template-columns:1fr 1fr}.directory a{font-size:9pt;padding:5px 0}.search-hit,.search-hit.current{background:transparent!important;outline:0!important}a{color:inherit;text-decoration:none}}
`;

const runtime = `
const sections = ${JSON.stringify(sections.map(s=>({id:s.id,title:s.title,headings:s.headings}))).replace(/</g,'\\u003c')};
const $ = id => document.getElementById(id);
const panels = [...document.querySelectorAll('.chapter')];
let activeSection = '', ticking = false;
function activate(id) {
  if(id===activeSection) return;
  activeSection=id; const s=sections.find(s=>s.id===id); if(!s)return;
  document.querySelectorAll('.chapter-link').forEach(a=>{const selected=a.hash==='#'+id;a.classList.toggle('active',selected);if(selected)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});
  $('toc-chapter').textContent=s.title;
  $('toc-links').replaceChildren(...s.headings.map(h=>{const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.text;a.className=h.depth===4?'sub':'';return a}));
  if(!s.headings.length){const p=document.createElement('p');p.textContent='本节可直接阅读。';$('toc-links').append(p)}
}
function onScroll(){if(ticking)return;ticking=true;requestAnimationFrame(()=>{ticking=false;let selected=panels[0];for(const p of panels){if(p.getBoundingClientRect().top<=150)selected=p;else break}activate(selected.dataset.section);const s=sections.find(s=>s.id===activeSection);let h=null;for(const item of s.headings){if($(item.id).getBoundingClientRect().top<=165)h=item;else break}document.querySelectorAll('#toc-links a').forEach(a=>a.classList.toggle('active',h&&a.hash==='#'+h.id));const available=document.documentElement.scrollHeight-innerHeight;$('progress').style.width=(available?Math.min(100,100*scrollY/available):0)+'%'})}
addEventListener('scroll',onScroll,{passive:true});
addEventListener('hashchange',()=>{const h=document.querySelector(':target');if(h){const chapter=h.closest('.chapter');if(chapter)activate(chapter.dataset.section)}});
const menu=$('mobile-menu');function closeMenu(){document.body.classList.remove('nav-open');menu.setAttribute('aria-expanded','false')}
menu.onclick=()=>{const open=document.body.classList.toggle('nav-open');menu.setAttribute('aria-expanded',String(open));if(open)document.querySelector('.sidebar a').focus()};
$('veil').onclick=()=>{closeMenu();menu.focus()};
document.querySelectorAll('.sidebar a').forEach(a=>a.addEventListener('click',()=>closeMenu()));
document.addEventListener('keydown',e=>{if(e.key==='Escape'){if(document.body.classList.contains('nav-open')){closeMenu();menu.focus()}else if($('search').value){$('search').value='';search('')}}});
$('print').onclick=()=>window.print();$('back-top').onclick=()=>{window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth'});$('page-title').focus({preventScroll:true})};
let fontSize=16;for(const [id,delta] of [['font-down',-1],['font-up',1]]){$(id).onclick=()=>{fontSize=Math.max(14,Math.min(20,fontSize+delta));document.documentElement.style.setProperty('--font',fontSize+'px');$('font-down').disabled=fontSize===14;$('font-up').disabled=fontSize===20}};
let hits=[],hitIndex=-1,timer,lastQuery='';
function clearHits(){document.querySelectorAll('mark.search-hit').forEach(m=>m.replaceWith(document.createTextNode(m.textContent)));$('reading').normalize();hits=[];hitIndex=-1}
function updateStatus(){const text=hits.length?(hitIndex+1)+' / '+hits.length:'无匹配';$('search-status').textContent=$('search').value.trim()?text:'';$('search-live').textContent=$('search').value.trim()?'搜索结果：'+text:'';$('match-tools').hidden=!hits.length}
function goHit(index){if(!hits.length)return;hits.forEach(m=>m.classList.remove('current'));hitIndex=(index+hits.length)%hits.length;const hit=hits[hitIndex];hit.classList.add('current');const details=hit.closest('details');if(details)details.open=true;hit.scrollIntoView({block:'center',behavior:'auto'});updateStatus()}
function search(query){lastQuery=query;clearHits();const q=query.trim().toLocaleLowerCase();if(!q){updateStatus();return}const walker=document.createTreeWalker($('reading'),NodeFilter.SHOW_TEXT,{acceptNode:n=>n.parentElement.closest('svg,script,style,.diagram-source,.anchor,summary')?NodeFilter.FILTER_REJECT:NodeFilter.FILTER_ACCEPT});const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);for(const node of nodes){const raw=node.nodeValue,low=raw.toLocaleLowerCase();let start=0,pos=low.indexOf(q),frag=document.createDocumentFragment();if(pos<0)continue;while(pos>=0){frag.append(document.createTextNode(raw.slice(start,pos)));const mark=document.createElement('mark');mark.className='search-hit';mark.textContent=raw.slice(pos,pos+q.length);frag.append(mark);hits.push(mark);start=pos+q.length;pos=low.indexOf(q,start)}frag.append(document.createTextNode(raw.slice(start)));node.replaceWith(frag)}if(hits.length)goHit(0);else updateStatus()}
$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>search($('search').value),180)});
$('search').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();clearTimeout(timer);if(lastQuery!==$('search').value||!hits.length)search($('search').value);else goHit(hitIndex+(e.shiftKey?-1:1))}});
$('prev-hit').onclick=()=>goHit(hitIndex-1);$('next-hit').onclick=()=>goHit(hitIndex+1);
$('page-title').setAttribute('tabindex','-1');activate('before-start');onScroll();
`;
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="肺炎相关电子病历公共卫生信号标注指南：病例判断、实体属性、关系、完整案例与质量控制。"><title>肺炎标注指南 · 阅读版</title><style>${css}</style></head>
<body><a class="skip" href="#reading">跳到正文</a><button class="veil" id="veil" aria-label="关闭目录"></button><div class="shell"><aside class="sidebar" id="sidebar" aria-label="章节导航"><a class="brand" href="#page-title"><span class="symbol" aria-hidden="true">＋</span><span><strong>肺炎标注手册</strong><small>ANNOTATION HANDBOOK</small></span></a><div class="nav-label">开始阅读</div><nav>${nav(['before-start','document-notes','contents'])}<div class="nav-label">病例判断与标注操作</div>${nav(numbered.slice(0,9).map(s=>s.id))}<div class="nav-label">流程、案例与质量</div>${nav(numbered.slice(9).map(s=>s.id))}<div class="nav-label">查阅与追溯</div>${nav(['appendix-a','appendix-b','references'])}</nav><div class="side-related">配套查阅<a href="label-dictionary.html">标签字典 · HTML 阅读版 ↗</a><a href="annotation_guide_v2.1.1.md">查看 Markdown 原稿 ↗</a><a href="../../label_studio/pneumonia_config.xml">当前统一页面配置 ↗</a><p>HTML 整理：2026-10-07<br>正文版本、状态以原稿为准</p></div></aside>
<div class="workspace"><header class="topbar"><button class="tool mobile-menu" id="mobile-menu" aria-controls="sidebar" aria-expanded="false">目录</button><span class="breadcrumbs">标注操作　/　主指南　/　阅读版</span><div class="tools"><div class="searchbox"><input id="search" type="search" aria-label="搜索指南全文" placeholder="搜索规则、案例或关键词…"><span class="search-status" id="search-status"></span><div id="match-tools" class="match-tools" hidden><button class="tool" id="prev-hit" aria-label="上一处匹配">↑</button><button class="tool" id="next-hit" aria-label="下一处匹配">↓</button></div></div><div class="text-size"><button class="tool" id="font-down" aria-label="缩小正文字号">A−</button><button class="tool" id="font-up" aria-label="放大正文字号">A＋</button></div><a class="tool source-button" href="annotation_guide_v2.1.1.md">原稿</a><button class="tool" id="print">打印 / PDF</button></div><div class="progress" id="progress" aria-hidden="true"></div></header><div class="search-live" id="search-live" role="status" aria-live="polite"></div>
<div class="layout"><main class="reading" id="reading"><div class="hero"><div class="eyebrow">标注工作指南 · 全文阅读</div>${hero}<div class="stats"><div><strong>16</strong>主体章节</div><div><strong>28</strong>完整案例</div><div><strong>4</strong>操作流程图</div></div><div class="quick-start"><a href="#chapter-3"><small>先判断病例</small>四分类与复核入口 ↗</a><a href="#chapter-11"><small>按步骤操作</small>每例流程与提交核对 ↗</a><a href="#chapter-12"><small>查阅示例</small>完整案例与边界处理 ↗</a></div></div><aside class="version-note unified" role="note"><strong>成人与儿童共用统一配置</strong>本指南与<a href="label-dictionary.html">标签字典</a>均已按<a href="../../label_studio/pneumonia_config.xml">pneumonia_config.xml</a>同步：49个实体标签、9个实体属性字段、5种关系及1个病例级单选控件。两类任务使用同一套页面操作，病例判断按第3—4章执行；儿童特有体征仍按其医学定义和纳入范围标注。</aside>${body}</main><aside class="toc" aria-label="本章目录"><div class="toc-label">本章速览</div><div id="toc-chapter" class="toc-chapter"></div><nav id="toc-links"></nav><div class="toc-note">点击标题旁的 # 可定位到该节。<br>搜索后可用 ↑ ↓ 或回车逐处查看。<br><br><a href="#appendix-a">快速判定卡 ↗</a><a href="label-dictionary.html">查看标签字典 ↗</a></div></aside></div><footer>来源：annotation_guide_v2.1.1.md · 保留源稿正文、表格、案例和引用 · 页面可离线阅读</footer></div></div><button class="back-top" id="back-top" aria-label="返回页首">↑</button><script>${runtime}</script></body></html>`;
fs.writeFileSync(outputPath,html,'utf8');
if (numbered.length!==16 || cases.length!==28 || diagrams.length!==4) throw new Error('Guide completeness check failed.');
console.log(JSON.stringify({output:outputPath, chapters:numbered.length, cases:cases.length, diagrams:diagrams.map(({source,...d})=>d), tables:(html.match(/<table>/g)||[]).length, headings:headings.length}, null, 2));
