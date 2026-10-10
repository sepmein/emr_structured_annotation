from pathlib import Path
import json
import tempfile
from playwright.sync_api import sync_playwright


BASE = Path('data/anotated_results/batch_1/review_2026-10-09_final').resolve()
DATA = json.loads((BASE/'audit.v2.json').read_text('utf-8'))
URL = (BASE/'review.html').as_uri()


with sync_playwright() as pw, tempfile.TemporaryDirectory(prefix='review-v2-test-') as temp:
    browser = pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
    page = browser.new_page(viewport={'width':1720,'height':1150})
    errors=[]
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(URL)
    assert page.locator('.case-item').count()==36
    assert page.locator('#metrics').inner_text().find('0.593')>=0
    assert page.locator('.mark').count()>0
    assert page.evaluate('attributesEqual({temporality:["current"],finding_context:["known_present"]},{finding_context:["known_present"],temporality:["current"]})')
    assert not page.locator('#overview').evaluate('(e)=>e.open')
    page.screenshot(path=str(Path('tmp/review_v2_overview.png').resolve()),full_page=False)

    # Matrix filters select the real two-person case disagreement, not the two files.
    page.locator('#overview summary').click()
    page.locator('[data-matrix-a="非目标"][data-matrix-b="待专业复核"]').click()
    assert page.locator('.case-item').count()==1
    assert 'T005' in page.locator('#case-heading').inner_text()
    page.locator('#clear-filters').click()
    page.locator('#overview summary').click()

    # On the first group B has extra 体温 on the same 发热 span; all labels are preserved.
    first_group=page.locator('#case-content tr[data-group]').first
    assert '多标签' in first_group.inner_text()
    first_group.click()
    assert '发热' in page.locator('#editor').inner_text()
    assert '220' in page.locator('#editor').inner_text()

    # IgG method mismatch is shown in Chinese and each side remains independently available.
    page.locator('[data-task="P230/LS41862/T093"]').click()
    page.locator('[data-tab=attributes]').click()
    assert '检测方式' in page.locator('#case-content').inner_text()
    assert '抗体滴度/双份' in page.locator('#case-content').inner_text()
    attr_row=next(r for r in DATA['review_rows'] if r['task']=='P230/LS41862/T093' and r['kind']=='属性' and r['A']['field']=='detection_method' and r['status']!='一致')
    page.locator('tr[data-row="'+attr_row['id']+'"]').click()
    assert '检测方式' in page.locator('.editor-title').inner_text()

    # Relationship graphs retain every source edge and the missing types.
    page.locator('[data-task="P230/LS41860/T068"]').click()
    page.locator('[data-tab=relations]').click()
    assert page.locator('.graph-panel svg').count()==2
    assert page.locator('.graph-edge').count()==9
    assert page.locator('.graph-edge path[stroke-dasharray]').count()==9
    page.locator('.graph-edge').first.click()
    assert '未记录类型' in page.locator('.editor-title').inner_text()
    page.locator('#case-main').scroll_into_view_if_needed()
    page.screenshot(path=str(Path('tmp/review_v2_relation_graph.png').resolve()),full_page=False)

    # All 49 labels can be inspected, including labels absent from this batch.
    page.locator('[data-nav=labels]').click()
    assert page.locator('#label-table tbody tr').count()==49
    page.locator('#label-table [data-label="肺炎支原体"]').click()
    assert page.locator('#label-filter').input_value()=='肺炎支原体'
    assert '肺炎支原体' in page.locator('#label-rule').inner_text()
    page.locator('#goto-label-cases').click()
    assert page.locator('#cases-view').is_visible()
    assert page.locator('.case-item').count()>0
    page.locator('#clear-filters').click()

    # Old IDs and source bindings survive export/import; selected answers never auto-sign.
    page.locator('[data-task="P221/LS41709/T290"]').click()
    page.locator('#edit-case').click()
    page.locator('[data-adopt=B]').click()
    assert page.locator('[data-field=final_answer]').input_value()=='非目标'
    assert page.locator('[data-field=status]').input_value()=='待裁决'
    for field,value in {'rule_ref':'指南§3.2 C3','reason':'UI smoke test','adjudicator':'TEST','reviewed_at':'2026-10-10','evidence_offsets':'[]'}.items():
        page.locator('[data-field='+field+']').fill(value)
    page.locator('[data-field=status]').select_option('已裁决')
    with page.expect_download() as dl:
        page.locator('#download').click()
    out=Path(temp)/'decisions.json'
    dl.value.save_as(out)
    saved=json.loads(out.read_text('utf-8'))
    assert len(saved['decisions'])==795
    assert {d['id'] for d in saved['decisions']}=={r['id'] for r in DATA['review_rows']}
    assert saved['sources']==DATA['sources']
    assert saved['review_context']==DATA['review_context']
    assert sum(d['status']=='已裁决' for d in saved['decisions'])==1
    page.evaluate('localStorage.clear()')
    page.reload()
    page.on('dialog',lambda dialog:dialog.accept())
    page.locator('#import').set_input_files(str(out))
    page.wait_for_function("document.getElementById('save').textContent.includes('已导入')")
    assert page.locator('#metrics').inner_text().find('1 / 795')>=0

    # Supplementary Unicode must not shift highlight offsets.
    result=page.evaluate('''() => {
      const e={id:'x',ref:'x',label:'发热',start:1,end:3,text:'发热',attributes:{}};
      const t={key:'FIXTURE',review_rows:[],annotations:{A:{entities:[e]},B:{entities:[{...e,id:'y',ref:'y'}]}}};
      return highlighted(t,'A',{start:0,end:4,text:'🙂发热。'},null);
    }''')
    assert '🙂' in result and '>发热</button>' in result

    # Narrow viewports reflow without document-level horizontal overflow.
    page.evaluate('localStorage.clear()')
    page.reload()
    for width in (1280,1024,760,390):
        page.set_viewport_size({'width':width,'height':950})
        page.locator('#clear-filters').click()
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,(width,page.evaluate('document.documentElement.scrollWidth'))
        page.locator('[data-task="P230/LS41860/T068"]').click()
        page.locator('[data-tab=relations]').click()
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1
        page.locator('[data-nav=labels]').click()
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1
        page.locator('[data-nav=cases]').click()
        if width==390:
            page.screenshot(path=str(Path('tmp/review_v2_mobile.png').resolve()),full_page=False)
    assert not errors,errors
    browser.close()
    print(json.dumps({'page_errors':errors,'matrix_filters':'PASS','entity_highlights':'PASS','attribute_comparison':'PASS',
        'actual_relation_graphs':'PASS','all_49_labels':'PASS','old_decision_ids_preserved':'PASS','draft_export_import':'PASS',
        'unicode_offsets':'PASS','responsive_widths':[1720,1280,1024,760,390]},ensure_ascii=False))
