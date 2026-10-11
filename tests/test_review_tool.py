"""Human review data integrity and persistence, using only fictional examples."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from emr_annotation.adjudication.review_tool import (
    group_case, initial_document, validate_document, ReviewStore, RevisionConflict, write)
from emr_annotation.annotation.schema import load_double_annotation_schema


class ReviewToolTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_double_annotation_schema(Path(__file__).resolve().parents[1]/'label_studio/pneumonia_config.xml')
        self.sample = {'task_id': 'synthetic_1', 'text': '🙂发热3天。', 'text_sha256': 'synthetic_hash',
                       'entities': [{'id': 'e1', 'start': 1, 'end': 3, 'text': '发热', 'label': '发热',
                                     'attributes': {'finding_context': 'known_present', 'temporality': 'current',
                                                    'experiencer': 'patient', 'conditional': 'no', 'hypothetical': 'no'}},
                                    {'id': 'e2', 'start': 3, 'end': 5, 'text': '3天', 'label': '时间表达', 'attributes': {}}],
                       'relations': [{'from_id': 'e1', 'to_id': 'e2', 'type': '持续时长', 'direction': 'right'}],
                       'case_decision': '信息不足', 'case_reason': '虚构流程测试', 'expert_signed': False}
        self.points = [
            {'id': 'p1', 'kind': 'labels', 'decision': '修订', 'original': {'start': 1, 'end': 5}},
            {'id': 'p2', 'kind': 'attribute_final', 'decision': '样例定值', 'final': {'entity_id': 'e1'}},
            {'id': 'p3', 'kind': 'relation', 'decision': '不沿用'},
            {'id': 'p4', 'kind': 'choices', 'decision': '保留', 'original': {'from_name': 'case_decision'}},
            {'id': 'p5', 'kind': 'labels', 'decision': '待复核', 'original': {'start': -1, 'end': 999}},
            {'id': 'p6', 'kind': 'entity_added', 'decision': '新增样例', 'final': self.sample['entities'][1]},
        ]
        self.case = {'sample': self.sample, 'points': self.points, 'annotations': [], 'alerts': []}

    def approved(self):
        doc = initial_document(self.case)
        doc.update(case_status='approved', reviewer='fictional_reviewer', new_entities_reviewed=True)
        doc['point_reviews'] = {p['id']: {'status': 'modified', 'note': '人工核对虚构原文'} for p in self.points}
        return doc

    def root(self, directory):
        root = Path(directory)
        (root/'cases').mkdir()
        write(root/'cases/synthetic_1.json', self.case)
        write(root/'index.json', {'bundle_id': 'synthetic_bundle', 'schema': self.schema,
                                 'cases': [{'task_id': 'synthetic_1'}]})
        return root

    def test_every_point_once_including_invalid_and_relation(self):
        groups = group_case(self.sample, self.points, [])
        ids = [i for g in groups for i in g['point_ids']]
        self.assertCountEqual(ids, [p['id'] for p in self.points])
        span = next(g for g in groups if g.get('start') == 1)
        self.assertEqual(span['end'], 5)
        self.assertIn('p5', next(g for g in groups if g['id'] == 'other')['point_ids'])

    def test_unicode_offsets_and_immutable_original(self):
        doc = initial_document(self.case)
        validate_document(doc, self.case, self.schema)
        doc['candidate']['entities'][0]['start'] = 2
        with self.assertRaisesRegex(ValueError, '原文不一致'):
            validate_document(doc, self.case, self.schema)
        doc = initial_document(self.case)
        doc['candidate']['text'] = 'changed'
        with self.assertRaisesRegex(ValueError, '不可修改'):
            validate_document(doc, self.case, self.schema)

    def test_review_completion_pending_and_required_attributes(self):
        doc = self.approved()
        validate_document(doc, self.case, self.schema)
        doc['point_reviews']['p5']['note'] = ''
        validate_document(doc, self.case, self.schema)
        doc = self.approved()
        del doc['point_reviews']['p1']
        with self.assertRaisesRegex(ValueError, '未审查'):
            validate_document(doc, self.case, self.schema)
        doc = self.approved()
        del doc['candidate']['entities'][0]['attributes']['experiencer']
        with self.assertRaisesRegex(ValueError, '必填属性'):
            validate_document(doc, self.case, self.schema)

    def test_deferred_draft_saves_without_reason_and_confirmation_is_separate(self):
        with tempfile.TemporaryDirectory() as d:
            store = ReviewStore(self.root(d))
            draft = initial_document(self.case)
            draft.update(reviewer='fictional_reviewer', case_status='reviewing')
            draft['point_reviews'] = {'p5': {'status': 'pending', 'note': ''}}
            store.save('synthetic_1', 0, draft)
            with self.assertRaisesRegex(ValueError, '未审查或暂缓'):
                store.approve('synthetic_1', 1)
            self.assertEqual(store.get('synthetic_1')['document'], draft)
            self.assertEqual(store.get('synthetic_1')['revision'], 1)
            self.assertEqual(len(store.history('synthetic_1')), 1)
            draft['point_reviews']['p5']['note'] = '虚构暂缓意见'
            store.save('synthetic_1', 1, draft)
            self.assertEqual(store.get('synthetic_1')['revision'], 2)
            self.assertEqual(store.export(False)['exported_cases'], 1)

    def test_approval_retains_revision_conflicts_and_accepts_checked_pending_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.root(d)
            store = ReviewStore(root)
            draft = self.approved()
            draft['case_status'] = 'reviewing'
            draft['point_reviews']['p5']['note'] = ''
            store.save('synthetic_1', 0, draft)
            with self.assertRaises(RevisionConflict):
                store.approve('synthetic_1', 0)
            result = store.approve('synthetic_1', 1)
            self.assertEqual(result['revision'], 2)
            self.assertEqual(store.get('synthetic_1')['document']['case_status'], 'approved')
            self.assertEqual(ReviewStore(root).summary()['synthetic_1']['reviewer'], 'fictional_reviewer')

    def test_relations_and_attribute_scope_rejected(self):
        doc = initial_document(self.case)
        doc['candidate']['relations'][0]['to_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, '关系类型'):
            validate_document(doc, self.case, self.schema)
        doc = initial_document(self.case)
        doc['candidate']['entities'][1]['attributes'] = {'finding_context': 'known_present'}
        with self.assertRaisesRegex(ValueError, '适用范围'):
            validate_document(doc, self.case, self.schema)

    def test_persistence_conflict_history_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.root(d)
            store = ReviewStore(root)
            original = (root/'cases/synthetic_1.json').read_bytes()
            doc = initial_document(self.case)
            self.assertEqual(store.save('synthetic_1', 0, doc)['revision'], 1)
            with self.assertRaises(RevisionConflict):
                store.save('synthetic_1', 0, self.approved())
            store.save('synthetic_1', 1, self.approved())
            reopened = ReviewStore(root)
            self.assertEqual(reopened.get('synthetic_1')['revision'], 2)
            self.assertEqual(len(reopened.history('synthetic_1')), 2)
            exported = reopened.export(True)
            self.assertEqual(exported['exported_cases'], 1)
            self.assertFalse(exported['references'][0]['expert_signed'])
            self.assertEqual(exported['references'][0]['reference_status'], 'human_reviewed_not_expert_gold')
            self.assertEqual(original, (root/'cases/synthetic_1.json').read_bytes())

    def test_restore_validates_all_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.root(d)
            store = ReviewStore(root)
            backup = {'format': 'emr_human_review_v1', 'bundle_id': 'synthetic_bundle',
                      'records': [{'task_id': 'synthetic_1', 'text_sha256': 'synthetic_hash', 'document': self.approved()}]}
            invalid = deepcopy(backup)
            invalid['records'].append({'task_id': 'missing', 'document': {}})
            with self.assertRaises(ValueError):
                store.restore(invalid)
            self.assertEqual(store.summary(), {})
            self.assertEqual(store.restore(backup)['imported'], 1)
            self.assertEqual(store.restore(backup)['unchanged'], 1)
            conflict = deepcopy(backup)
            conflict['records'][0]['document']['case_note'] = 'different'
            with self.assertRaises(RevisionConflict):
                store.restore(conflict)


if __name__ == '__main__':
    unittest.main()
