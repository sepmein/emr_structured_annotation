"""Synthetic checks for provisional supervision and patient separation."""
import json
from pathlib import Path
import tempfile
import unittest

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.training_data.schema_multitask import prepare, project_typed_relation


SCHEMA = Path(__file__).resolve().parents[1] / 'label_studio/pneumonia_config.xml'


class SchemaPreparationTests(unittest.TestCase):
    def test_typed_multilabel_region_has_only_unique_schema_projection(self):
        schema = load_double_annotation_schema(SCHEMA)
        regions = {'source':[{'id':'fever','label':'发热'}],
                   'target':[{'id':'wrong','label':'发热'},{'id':'time','label':'时间表达'}]}
        relation = {'ref':'synthetic/raw/R1','from_id':'source','to_id':'target',
                    'direction':'right','labels':['持续时长']}
        projected = project_typed_relation(relation,regions,schema)
        self.assertEqual((projected['from_id'],projected['to_id'],projected['type']),('fever','time','持续时长'))
        self.assertEqual(projected['source_relation_ref'],'synthetic/raw/R1')
        self.assertIsNone(project_typed_relation({**relation,'labels':[]},regions,schema))
        ambiguous = {**regions,'target':regions['target']+[{'id':'time2','label':'时间表达'}]}
        self.assertIsNone(project_typed_relation(relation,ambiguous,schema))
        reversed_relation={**relation,'from_id':'target','to_id':'source','direction':'left'}
        self.assertEqual(project_typed_relation(reversed_relation,regions,schema)['from_id'],'fever')

    def fixture(self, directory, invalid=False):
        controls = load_double_annotation_schema(SCHEMA)['controls']
        control = next(c for c, labels in controls.items() if '发热' in labels)
        tasks = []
        for i in range(6):
            text = f'发热。虚构样本{i}。'
            results = [{'type':'labels','id':'e1','from_name':control,'to_name':'chief_complaint_text',
                        'value':{'start':0,'end':2,'text':'发热','labels':['发热']}},
                       {'type':'choices','from_name':'case_decision','to_name':'chief_complaint_text',
                        'value':{'choices':['信息不足']}}]
            if invalid:
                results[0]['value']['text'] = '错误选区'
                results.append({'type':'relation','from_id':'e1','to_id':'e1','direction':'right'})
            tasks.append({'id':i,'project':1,'data':{'text':text,'patient_group_id':f'synthetic_patient_{i//2}'},
                          'annotations':[{'id':i*2+j,'completed_by':j+1,'result':results} for j in range(2)]})
        (directory / 'project-1.json').write_text(json.dumps(tasks,ensure_ascii=False),encoding='utf-8')

    def test_both_views_and_patient_components_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            audit = prepare(root, SCHEMA, root/'prepared')
            self.assertEqual(audit['annotation_views'],12)
            self.assertTrue(audit['source_hashes_unchanged'])
            seen = {}
            for split in ('train','validation','test'):
                rows = json.loads((root/'prepared'/f'{split}.json').read_text('utf-8'))
                for row in rows:
                    patient = row['patient_id']
                    self.assertEqual(seen.setdefault(patient,split),split)
                    self.assertEqual(row['entities'][0]['attributes'],{})
                    self.assertFalse(row['relations_supervised'])

    def test_bad_spans_and_untyped_relations_are_masked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.fixture(root,invalid=True)
            audit=prepare(root,SCHEMA,root/'prepared')
            rows=json.loads((root/'prepared'/'reference.json').read_text('utf-8'))
            self.assertTrue(all(not r['ner_supervised'] and not r['relations_supervised'] for r in rows))
            self.assertEqual(audit['exclusions']['invalid_relations'],12)
            self.assertEqual(audit['exclusions']['invalid_entities'],12)


if __name__ == '__main__':
    unittest.main()
