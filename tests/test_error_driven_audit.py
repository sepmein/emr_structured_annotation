import unittest
from copy import deepcopy
from emr_annotation.evaluation.model_error_audit import inspect,render
from emr_annotation.evaluation.explicit_evidence_binding import bind,explicit_attributes,anatomical_conflict
from emr_annotation.evaluation.clinical_focus import focus

def schema():
    return dict(labels=['病原甲','数值','体温'],relations=[],fields={
        'test_result':dict(per_region=True,scope=['病原甲'],options=['positive','negative','unknown']),
        'specimen_type':dict(per_region=True,scope=['病原甲'],options=['痰','原文未说明']),
        'detection_method':dict(per_region=True,scope=['病原甲'],options=['核酸检测','抗体IgM','原文未说明']),
        'case_decision':dict(per_region=False,scope=[],options=['非目标'])})

def entity(text,word,label='病原甲',attrs=None,ident='e'):
    start=text.index(word)
    return dict(id=ident,start=start,end=start+len(word),label=label,attributes=attrs or {})

def row(text,entities):return dict(task_id='fictitious',text=text,entities=entities,relations=[],case_decision='非目标')

class ErrorDrivenAuditTests(unittest.TestCase):
    def test_duplicate_predictions_preserve_error_counts(self):
        t='病原甲';e=entity(t,t);p=deepcopy(e);p['id']='dup'
        summary,errors,cases=inspect([row(t,[e])],[row(t,[e,p])],schema())
        self.assertEqual(summary['entities']['overall']['micro']['fp'],1)
        self.assertEqual([r['category'] for r in errors if r['domain']=='entity'],['duplicate_prediction'])

    def test_unknown_slots_excluded_and_upstream_separate(self):
        t='病原甲';e=entity(t,t,attrs={'test_result':'positive'})
        summary,errors,_=inspect([row(t,[e])],[row(t,[])],schema())
        self.assertNotIn('specimen_type',summary['attributes'])
        self.assertEqual(summary['attributes']['test_result']['upstream_entity_error'],1)
        self.assertEqual(summary['attributes']['test_result']['matched_attribute_error'],0)

    def test_exact_text_and_case_coverage_required(self):
        with self.assertRaises(ValueError):inspect([row('病原甲',[])],[row('其他文本',[])],schema())
        c=row('病原甲',[]);c['task_id']='extra'
        with self.assertRaises(ValueError):inspect([row('病原甲',[])],[row('病原甲',[])],schema(),case_predictions=[c])

    def test_result_does_not_cross_neighbor_negative(self):
        t='病原甲抗原阳性，病原乙阴性。';values,_=explicit_attributes(t,entity(t,'病原甲'),schema())
        self.assertEqual(values['test_result'],'positive')

    def test_structured_result_excludes_reference_interval(self):
        t='检验项目名称：病原甲RNA\n检验结果：阳性\n参考区间：阴性\n\n操作时间：某日'
        values,ev=explicit_attributes(t,entity(t,'病原甲'),schema())
        self.assertEqual(values['test_result'],'positive')
        self.assertEqual(t[ev['test_result']['start']:ev['test_result']['end']],'阳性')
        t='检验项目名称：病原甲\n检验结果：0.2\n参考区间：阴性'
        self.assertNotIn('test_result',explicit_attributes(t,entity(t,'病原甲'),schema())[0])

    def test_invalid_schema_method_not_emitted(self):
        t='病原甲IgG阳性。'
        self.assertNotIn('detection_method',explicit_attributes(t,entity(t,'病原甲'),schema())[0])

    def test_anatomical_levels_preserve_real_temperature_and_reference(self):
        t='胸椎T9-10处改变，体温T38.5℃。'
        e=entity(t,'T9','体温');real=entity(t,'T38.5','体温',ident='real')
        self.assertTrue(anatomical_conflict(t,e));self.assertFalse(anatomical_conflict(t,real))
        original=[row(t,[e,real])];snapshot=deepcopy(original)
        self.assertEqual(len(bind(original,schema(),guard=True)[0]['entities']),1)
        self.assertEqual(original,snapshot)

    def test_html_embedded_text_and_title_are_escaped(self):
        t='</script><img src=x onerror=alert(1)>'
        summary,errors,cases=inspect([row(t,[])],[row(t,[])],schema())
        page=render(summary,[dict(error_id='x',context=t)],cases,'<script>')
        self.assertNotIn(t,page);self.assertIn('&lt;script&gt;',page)
        self.assertIn('page*50',page)

    def test_focus_preserves_negation_and_time(self):
        t='既往无咳嗽。近日肺部复查未见浸润。'
        self.assertIn('既往无咳嗽',focus(t));self.assertIn('未见浸润',focus(t))

if __name__=='__main__':unittest.main()
