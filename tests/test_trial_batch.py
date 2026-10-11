"""Synthetic tests for export integration and comparison contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from emr_annotation.annotation_analysis.trial_batch import (integrate, analyse,
    attribute_statistics, load_schema, write_csv, run)
from emr_annotation.annotation_analysis.double_annotations import build_report
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def task(task_id=1, project=228, users=(92,95)):
    return {'id':task_id,'project':project,'data':{'text':'发热','task_id':f'T{task_id}'},
        'annotations':[{'id':task_id*100+i,'completed_by':user,'was_cancelled':False,
            'result':[{'id':'case','type':'choices','from_name':'case_decision',
                       'to_name':'chief_complaint_text','value':{'choices':['信息不足']}}]} for i,user in enumerate(users)]}


class TrialBatchTests(unittest.TestCase):
    def integrate_pair(self, a, b):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'project-228-old.json';q=Path(d)/'project-228-new.json'
            p.write_text(json.dumps([a]),'utf-8');q.write_text(json.dumps([b]),'utf-8')
            return integrate([p,q])

    def test_duplicate_submission_ignores_draft_only_changes(self):
        a=task();b=copy.deepcopy(a);b['drafts']=[{'result':[]}]
        selected,sources,lineage,blocked=self.integrate_pair(a,b)
        self.assertEqual(len(selected),1);self.assertEqual(len(sources),2)
        self.assertEqual(lineage[0]['metadata_difference_keys'],['drafts'])
        self.assertEqual(lineage[0]['version_count'],2);self.assertFalse(blocked)

    def test_conflicting_submission_is_blocked_not_latest_selected(self):
        a=task();b=copy.deepcopy(a);b['annotations'][0]['result'][0]['value']['choices']=['非目标']
        selected,_,lineage,blocked=self.integrate_pair(a,b)
        self.assertFalse(selected);self.assertEqual(len(blocked),1)
        self.assertEqual(lineage[0]['status'],'blocked_conflicting_submitted_content')

    def test_changed_original_text_and_same_person_pair_are_blocked(self):
        a=task();b=copy.deepcopy(a);b['data']['text']='发热。'
        self.assertFalse(self.integrate_pair(a,b)[0])
        same=task(users=(92,92));self.assertFalse(self.integrate_pair(same,same)[0])

    def test_matching_task_number_in_different_projects_is_not_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';q=Path(d)/'b.json'
            p.write_text(json.dumps([task(project=220)]),'utf-8')
            q.write_text(json.dumps([task(project=228)]),'utf-8')
            self.assertEqual(len(integrate([p,q])[0]),2)

    def test_two_pair_groups_keep_real_project_and_identity(self):
        schema=load_schema(ROOT/'label_studio/pneumonia_config.xml')
        p,tables=analyse([task(1,users=(92,95)),task(2,users=(1,92))],{'228':schema},{})
        self.assertEqual(set(p['detailed']['pair_groups']),{'228:92/95','228:1/92'})
        self.assertEqual(p['summary']['tasks'],2)
        self.assertIsNone(p['summary']['projects']['228']['case']['kappa'])
        self.assertIsNone(p['detailed']['pair_groups']['228:1/92']['case']['kappa'])
        self.assertTrue(all(t['project']=='228' and t['key'].startswith('P228/') for t in p['tasks']))
        self.assertEqual(len({r['id'] for r in p['review_rows']}),len(p['review_rows']))
        self.assertEqual(len(tables['annotations']),4)
        self.assertTrue(all(l['f1'] is None for l in p['labels']))

    def test_blank_exclusion_does_not_remove_one_sided_missing_values(self):
        s=attribute_statistics(Counter(matched_slots=70,equal_nonempty=4,both_blank=50,one_blank=14,different_nonempty=2))
        self.assertEqual(s['nonblank_slots'],20)
        self.assertEqual(s['nonblank_agreement'],.2)
        self.assertAlmostEqual(s['blank_inclusive_agreement'],54/70)
        self.assertIsNone(attribute_statistics(Counter(matched_slots=3,both_blank=3))['nonblank_agreement'])

    def test_missing_relation_type_is_not_evaluated_as_zero_or_inferred(self):
        schema=load_schema(ROOT/'label_studio/pneumonia_config.xml');t=task()
        for a in t['annotations']:
            a['result'] += [{'id':'x','type':'labels','from_name':'symptons_labels',
                'to_name':'chief_complaint_text','value':{'start':0,'end':2,'text':'发热','labels':['发热']}},
                {'type':'relation','from_id':'x','to_id':'x','direction':'right'}]
        p,_=analyse([t],{'228':schema},{})
        self.assertEqual(p['summary']['relations_not_fully_typed'],2)
        self.assertIsNone(p['summary']['relations'])
        self.assertTrue(all(r['full_f1'] is None for r in p['detailed']['relations']))
        self.assertEqual(p['tasks'][0]['annotations']['A']['relations'][0]['labels'],[])

    def test_csv_source_text_cannot_become_spreadsheet_formula(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'text.csv';write_csv(p,[{'source':'=1+1','count':3}])
            self.assertIn("'=1+1",p.read_text('utf-8-sig'))

    def test_report_handles_unassessable_typed_relation_metric(self):
        schema=load_schema(ROOT/'label_studio/pneumonia_config.xml');t=task()
        t['annotations'][0]['result'].append({'type':'relation','from_id':'absent','to_id':'absent'})
        p,_=analyse([t],{'228':schema},{})
        p['limitations']=[]
        self.assertIn('暂不可评价',build_report(p))

    def test_full_pipeline_keeps_lineage_and_missing_fields_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);inputs=folder/'input';inputs.mkdir();output=folder/'output'
            rows=[task(1),task(2,users=(1,92))]
            (inputs/'project-228-old.json').write_text(json.dumps(rows),'utf-8')
            newer=copy.deepcopy(rows);newer[0]['drafts']=[]
            (inputs/'project-228-new.json').write_text(json.dumps(newer),'utf-8')
            args=(inputs,output,'run_test',None,ROOT/'label_studio/pneumonia_config.xml',
                  ROOT/'documentation/annotation/annotation_guide_v2.1.1.md',ROOT/'documentation/annotation/label_dictionary_v2.1.1.md')
            result=run(*args)
            self.assertEqual(result['integration']['removed_duplicate_task_rows'],2)
            self.assertEqual(result['summary']['tasks'],2)
            self.assertIsNone(result['summary']['entities']['symmetric_f1'])
            stats=json.loads((output/'annotation_analysis/run_test/statistics.json').read_text('utf-8'))
            self.assertEqual(len(stats['detailed']['group_relations']),2)
            self.assertFalse(stats['schemas'][0]['verified'])
            manifest=json.loads((output/'delivery/run_test/manifest.json').read_text('utf-8'))
            self.assertTrue(manifest['validation']['source_hashes_unchanged'])
            self.assertEqual(len(manifest['sources']),2)
            self.assertIn('statistics.html',(output/'delivery/run_test/index.html').read_text('utf-8'))
            with self.assertRaises(ValueError):run(*args)


if __name__=='__main__':unittest.main()
