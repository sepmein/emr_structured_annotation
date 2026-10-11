"""Fictional span boundaries and suppression contracts; no model required."""
import unittest
from emr_annotation.training_data.span_alignment import align_entity,suppress_redundant_spans


class SpanAlignmentTests(unittest.TestCase):
    def test_inside_token_targets_restore_exact_characters(self):
        offsets=[(0,0),(0,3),(3,5),(0,0)]
        self.assertIsNone(align_entity(offsets,1,4,4))
        self.assertEqual(align_entity(offsets,1,4,4,True),(1,3,1,1))
        b,e,left,right=align_entity(offsets,1,4,4,True)
        self.assertEqual((offsets[b][0]+left,offsets[e-1][1]-right),(1,4))

    def test_partial_long_and_invalid_trim_not_represented(self):
        offsets=[(4,7),(7,9)]
        self.assertIsNone(align_entity(offsets,3,8,4,True))
        self.assertIsNone(align_entity(offsets,5,8,1,True))
        self.assertIsNone(align_entity(offsets,6,8,4,True,max_trim=1))

    def test_duplicate_unicode_token_offsets_use_whole_envelope(self):
        self.assertEqual(align_entity([(0,0),(0,1),(0,1),(1,2)],0,1,4,True),(1,3,0,0))

    def test_same_label_suppression_keeps_different_labels_and_adjacent_mentions(self):
        records=[dict(start=0,end=4,label='label_a',score=.9),dict(start=1,end=3,label='label_a',score=.8),dict(start=1,end=3,label='label_b',score=.8),dict(start=4,end=5,label='label_a',score=.7)]
        result=suppress_redundant_spans(records,.8)
        self.assertEqual(len(result),3)
        self.assertEqual(suppress_redundant_spans(records,0),records)


if __name__=='__main__':unittest.main()
