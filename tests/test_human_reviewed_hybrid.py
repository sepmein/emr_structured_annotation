"""Synthetic-only contracts for generation, evidence and supervised scope."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.training_data.human_reviewed import generate,prepare,Builder
from emr_annotation.evaluation.clinical_evidence import extract,clinical_state,project_case_cues,CASE_CUE_NAMES

ROOT=Path(__file__).resolve().parents[1]


class HybridDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema=load_double_annotation_schema(ROOT/'label_studio/pneumonia_config.xml')

    def test_thousand_deterministic_valid_records_and_full_coverage(self):
        rows=generate(self.schema)
        self.assertEqual(rows,generate(self.schema))
        self.assertEqual(len({r['text'] for r in rows}),1000)
        self.assertEqual({e['label'] for r in rows for e in r['entities']},set(self.schema['labels']))
        self.assertEqual({x['type'] for r in rows for x in r['relations']},set(self.schema['relations']))
        self.assertEqual({r['case_decision'] for r in rows},set(self.schema['fields']['case_decision']['values']))
        self.assertTrue(all(r['synthetic'] and not r['expert_signed'] for r in rows))
        self.assertTrue(any(e.get('attributes',{}).get('hypothetical')=='yes' for r in rows for e in r['entities']))
        self.assertTrue(any(e.get('attributes',{}).get('conditional')=='yes' for r in rows for e in r['entities']))
        for r in rows:
            for e in r['entities']:self.assertEqual(e['text'],r['text'][e['start']:e['end']])
            for rel in r['relations']:
                if rel['type']=='结束时间':
                    source=next(e for e in r['entities'] if e['id']==rel['from_id'])
                    self.assertEqual(source['label'],'发热')
                    self.assertIn('该症状结束于',r['text'])

    def test_negative_historical_planned_and_nonpatient_not_active_evidence(self):
        for attrs in [dict(finding_context='known_absent'),dict(temporality='historical'),dict(hypothetical='yes'),dict(experiencer='family')]:
            b=Builder(self.schema);b.add('肺炎','肺炎诊断',**attrs)
            fact=extract(b.text,b.entities,[],self.schema)
            self.assertEqual(fact['syndromes'][1],0)
            self.assertNotEqual(clinical_state(b.entities[0]),'current_present')

    def test_unknown_test_is_not_negative_and_measurement_requires_edge(self):
        b=Builder(self.schema);a=b.add('体温','体温');v=b.add('38.5','数值');b.add('流感病毒','流感病毒')
        fact=extract(b.text,b.entities,[],self.schema)
        self.assertEqual(fact['pathogen_mask'],[0,0,0,0])
        self.assertEqual(fact['numeric_evidence'],{})
        b.edge(a,v,'测量');fact=extract(b.text,b.entities,b.relations,self.schema)
        self.assertEqual(fact['numeric_evidence']['measurement:体温:available'],1.)
        old=Builder(self.schema);old.add('既往');a=old.add('体温','体温');v=old.add('38.5','数值');old.edge(a,v,'测量')
        self.assertEqual(extract(old.text,old.entities,old.relations,self.schema)['numeric_evidence'],{})
        b.entities[-1]['attributes']['test_result']='negative'
        fact=extract(b.text,b.entities,b.relations,self.schema)
        self.assertEqual(fact['pathogens'][0],0);self.assertEqual(fact['pathogen_mask'][0],1)

    def test_human_status_and_original_split_identity_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);prior=base/'prior';prior.mkdir()
            rows=generate(self.schema,49)
            r=deepcopy(rows[0]);r.update(synthetic=False,reference_status='human_reviewed_not_expert_gold',human_reviewer='fictional_reviewer')
            ref=base/'reference.json';ref.write_text(json.dumps([r],ensure_ascii=False),'utf-8')
            for n in ('train','validation','test'):(prior/(n+'.json')).write_text(json.dumps([r] if n=='train' else [],ensure_ascii=False),'utf-8')
            result=prepare(ref,ROOT/'label_studio/pneumonia_config.xml',prior,base/'prepared',49)
            self.assertEqual(result['real_split_counts'],{'train':1,'validation':0,'test':0})
            self.assertEqual(result['synthetic_cases'],49)
            r['patient_id']='changed_patient';ref.write_text(json.dumps([r],ensure_ascii=False),'utf-8')
            with self.assertRaisesRegex(ValueError,'patient/text'):prepare(ref,ROOT/'label_studio/pneumonia_config.xml',prior,base/'invalid',49)
            self.assertFalse((base/'invalid').exists())

    def test_new_clis_help_has_no_model_imports(self):
        for name in ('train_hybrid.py','predict_hybrid.py','evaluate_hybrid.py','train_small_case.py','evaluate_modular.py','predict_modular.py'):
            result=subprocess.run([sys.executable,'-S',str(ROOT/'modernbert_ml_backend'/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_case_wording_negation_does_not_cross_clause(self):
        values=dict(zip(CASE_CUE_NAMES,project_case_cues('无发热，咳嗽。否认胸闷。')))
        self.assertEqual(values['respiratory_mention'],2/5)
        self.assertEqual(values['respiratory_negated'],1/5)
        self.assertEqual(values['respiratory_historical'],0)

    def test_case_wording_contract_and_explicit_gaps(self):
        values=project_case_cues('既往咳嗽。关键附件未提供，拟复查。')
        self.assertEqual(len(values),len(CASE_CUE_NAMES))
        named=dict(zip(CASE_CUE_NAMES,values))
        self.assertEqual(named['respiratory_historical'],1/5)
        self.assertGreater(named['unresolved_cross_reference'],0)
        self.assertEqual(named['explicit_possible_pneumonia'],0)


if __name__=='__main__':unittest.main()
