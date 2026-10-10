import unittest
from pathlib import Path

from emr_annotation.annotation_analysis.double_annotations import build_html, compare_task, display_windows, kappa, load_schema, metric, parse_annotation


ROOT = Path(__file__).resolve().parents[1]


def entity(rid, start, end, text, label, control='measure_entities'):
    return dict(id=rid, type='labels', from_name=control, to_name='chief_complaint_text',
                value=dict(start=start, end=end, text=text, labels=[label]))


class DoubleAnnotationAuditTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(ROOT/'label_studio/pneumonia_config.xml')

    def parse(self, results, side='A', text='T38℃'):
        case = dict(id='case', type='choices', from_name='case_decision',
                    to_name='chief_complaint_text', value=dict(choices=['非目标']))
        return parse_annotation(dict(data=dict(text=text)),dict(id=1,completed_by=1,
                                result=[case]+results),self.schema,side,'P1/LS1/T1')

    def test_schema_descendants_and_aliases(self):
        global_schema = load_schema(ROOT/'label_studio/pneumonia_config.global-single.xml')
        self.assertEqual(set(global_schema['labels']),set(self.schema['labels']))
        self.assertEqual(len(global_schema['labels']),49)
        self.assertEqual(self.schema['fields']['finding_context']['aliases']['明确存在'],'known_present')

    def test_missing_relation_type_is_not_inferred_and_reverse_topology_is_flagged(self):
        results=[entity('t',0,1,'T','体温'),entity('n',1,3,'38','数值','measure_labels'),
                 dict(type='relation',from_id='n',to_id='t',direction='right')]
        parsed=self.parse(results)
        codes={i['code'] for i in parsed['issues']}
        self.assertIn('relation_type_missing_or_invalid',codes)
        self.assertIn('relation_topology_candidate',codes)
        self.assertEqual(parsed['relations'][0]['labels'],[])

    def test_region_uuids_do_not_determine_agreement(self):
        a=self.parse([entity('alpha',0,1,'T','体温')])
        b=self.parse([entity('beta',0,1,'T','体温')],'B')
        rows,_=compare_task(a,b,self.schema)
        self.assertEqual([r['status'] for r in rows if r['kind']=='实体'],['一致'])

    def test_shared_missing_attributes_are_not_filled_agreement(self):
        a=self.parse([entity('a',0,2,'发热','发热','symptons_labels')],text='发热')
        b=self.parse([entity('b',0,2,'发热','发热','symptons_labels')],'B',text='发热')
        rows,stats=compare_task(a,b,self.schema)
        self.assertEqual(stats['finding_context']['both_blank'],1)
        self.assertEqual(stats['finding_context']['equal_nonempty'],0)
        self.assertIn('双方漏填必填',[r['status'] for r in rows])

    def test_multilabel_region_and_exact_text_mismatch_are_preserved(self):
        parsed=self.parse([entity('a',0,1,'T','体温'),entity('a',0,1,'T','数值','measure_labels'),
                           entity('b',1,3,'3','数值','measure_labels')])
        codes={i['code'] for i in parsed['issues']}
        self.assertIn('multi_label_region',codes)
        self.assertIn('span_text_mismatch',codes)
        self.assertEqual(len(parsed['entities']),3)
        self.assertEqual(len(parsed['inventory']),4)

    def test_left_direction_normalizes_and_self_edge_is_reported(self):
        parsed=self.parse([entity('t',0,1,'T','体温'),entity('n',1,3,'38','数值','measure_labels'),
                           dict(type='relation',from_id='n',to_id='t',direction='left',labels=['测量']),
                           dict(type='relation',from_id='t',to_id='t',direction='right')])
        self.assertEqual(parsed['relations'][0]['source'],[(0,1,'体温')])
        self.assertIn('self_relation',{i['code'] for i in parsed['issues']})

    def test_duplicates_and_empty_metrics(self):
        self.assertEqual(metric(['x','x'],['x'])['symmetric_f1'],0.666667)
        self.assertIsNone(metric([],[])['symmetric_f1'])
        self.assertIsNone(kappa([('非目标','非目标')])['kappa'])

    def test_display_windows_mask_identifiers_without_moving_offsets(self):
        text='姓名：测试姓名\n电话：13800138000\n体温T38℃'
        start=text.index('T')
        windows=display_windows(text,[(start,start+1)],padding=100)
        self.assertEqual(len(windows),1)
        masked=windows[0]['text']
        self.assertEqual(len(masked),len(text))
        self.assertEqual(masked[start:start+4],'T38℃')
        self.assertNotIn('测试姓名',masked)
        self.assertNotIn('13800138000',masked)

    def test_display_windows_merge_and_retain_non_bmp_character_coordinates(self):
        text='🙂发热'+('甲'*150)+'体温38℃'
        windows=display_windows(text,[(1,3),(153,155)],padding=5)
        self.assertEqual(len(windows),2)
        self.assertEqual(windows[0]['text'][1:3],'发热')
        self.assertEqual(windows[1]['text'][153-windows[1]['start']:155-windows[1]['start']],'体温')
        joined=display_windows('发热，体温38℃',[(0,2),(3,5)],padding=3)
        self.assertEqual(len(joined),1)

    def test_page_payload_is_escaped_and_has_no_remote_dependency(self):
        page=build_html({'fixture':'</script><script>unsafe</script>'})
        self.assertIn('\\u003c/script\\u003e',page)
        self.assertNotIn('src="http',page)
        self.assertIn('关系方向图',page)


if __name__=='__main__':unittest.main()
