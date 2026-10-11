"""Synthetic audit and freeze invariants; no patient text or ML runtime."""
import json
from pathlib import Path
import tempfile
import unittest

from emr_annotation.adjudication.ai_sample import propose, build_point_ledger, validate_samples, to_label_studio, build_review_html
from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.annotation_analysis.double_annotations import parse_annotation
from emr_annotation.training_data.ai_sample import prepare

SCHEMA = Path(__file__).resolve().parents[1]/'label_studio/pneumonia_config.xml'


class AISampleTests(unittest.TestCase):
    def setUp(self):
        self.schema=load_double_annotation_schema(SCHEMA)

    def task(self,text,results=None):
        return {'id':1,'project':1,'data':{'text':text,'patient_group_id':'synthetic_patient'},
                'annotations':[{'id':1,'completed_by':1,'result':results or []}]}

    def test_cause_negation_does_not_negate_fever(self):
        entities,*_=propose(self.task('患者无明显诱因出现发热，否认喘息。'),self.schema)
        attrs={e['label']:e['attributes'] for e in entities}
        self.assertEqual(attrs['发热']['finding_context'],'known_present')
        self.assertEqual(attrs['喘鸣喘息']['finding_context'],'known_absent')

    def test_objective_report_not_negated_by_previous_symptom(self):
        entities,*_=propose(self.task('无胸痛咯血，胸部CT：两肺炎症，为诊治拟“肺炎”收入院。'),self.schema)
        self.assertEqual([(e['label'],e['attributes']['finding_context']) for e in entities],
                         [('肺炎影像','known_present'),('肺炎诊断','known_present')])

    def test_reference_range_not_patient_negative(self):
        t='院内检验项目名称：肺炎支原体IgM\n院内检验定性结果名称：阳性\n参考区间：阴性'
        entities,*_=propose(self.task(t),self.schema)
        self.assertEqual(entities[0]['attributes']['test_result'],'positive')
        self.assertEqual(entities[0]['attributes']['specimen_type'],'原文未说明')

    def test_negative_is_not_borrowed_across_test_groups(self):
        entities,*_=propose(self.task('肺炎支原体IgM待回报；腺病毒抗原阴性。'),self.schema)
        index={e['label']:e['attributes']['test_result'] for e in entities}
        self.assertEqual(index,{'肺炎支原体':'unknown','腺病毒':'negative'})

    def test_rest_oxygen_default_and_exercise_exclusion(self):
        entities,rels,*_=propose(self.task('SpO2 92%。步行后SpO2 88%。体温38℃。'),self.schema)
        self.assertEqual(sum(e['label']=='静息状态指氧饱和度' for e in entities),1)
        self.assertNotIn('发热',[e['label'] for e in entities])
        self.assertTrue(all(r['type']=='测量' for r in rels))

    def test_every_raw_result_traced_and_unknown_masks_loss(self):
        control=next(c for c,labels in self.schema['controls'].items() if '肺炎影像' in labels)
        task=self.task('虚构语境甲。',[{'id':'x','type':'labels','from_name':control,'to_name':'chief_complaint_text',
                                       'value':{'start':0,'end':5,'text':'虚构语境甲','labels':['肺炎影像']}}])
        sample={'task_id':'p1_t1','text':task['data']['text'],'entities':[],'relations':[],'ner_supervised':True,'case_decision':'信息不足','case_rule':'3.2D','case_reason':'语境不足'}
        ledger,counts=build_point_ledger([task],[sample],self.schema)
        self.assertEqual(counts['original_results'],1)
        self.assertEqual(len(ledger),1)
        self.assertFalse(sample['ner_supervised'])

    def test_export_roundtrip_is_prediction_not_signed_annotation(self):
        entities,rels,issues,_=propose(self.task('无发热。T 37℃。'),self.schema)
        sample={'task_id':'p1_t1','text':'无发热。T 37℃。','entities':entities,'relations':rels,'issues':issues,
                'case_decision':'非目标','reference_status':'ai_sample_not_expert_gold'}
        validate_samples([sample],self.schema)
        exported=to_label_studio([sample],self.schema)[0]
        self.assertNotIn('annotations',exported)
        parsed=parse_annotation({'data':exported['data']},exported['predictions'][0],self.schema,'AI','synthetic')
        self.assertFalse(parsed['issues'])

    def test_split_preserved_and_changed_patient_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prior=root/'prior';prior.mkdir();records=[]
            for i,name in enumerate(('train','validation','test')):
                r={'task_id':f'p1_t{i}','project':1,'source_task_id':i,'patient_id':f'synthetic_{i}',
                   'text':f'虚构正常查体{i}。','text_sha256':'synthetic','entities':[],'relations':[],
                   'case_decision':'非目标','case_reviewed_by_ai':True,'reference_status':'ai_sample_not_expert_gold',
                   'ner_supervised':True,'issues':[]}
                records.append(r)
                (prior/(name+'.json')).write_text(json.dumps([r,{**r,'task_id':r['task_id']+'_a2'}]),'utf-8')
            source=root/'samples.json';source.write_text(json.dumps(records),'utf-8')
            audit=prepare(source,SCHEMA,root/'out',prior)
            self.assertEqual(audit['split_counts'],{'train':1,'validation':1,'test':1})
            records[0]['patient_id']='different_patient'
            source.write_text(json.dumps(records),'utf-8')
            with self.assertRaisesRegex(ValueError,'Changed text or patient'):
                prepare(source,SCHEMA,root/'out2',prior)

    def test_untrusted_text_cannot_end_embedded_json_script(self):
        content=build_review_html([{'text':'</script><script>bad()</script>'}],[],{})
        self.assertNotIn('</script><script>bad()',content)


if __name__=='__main__':
    unittest.main()
