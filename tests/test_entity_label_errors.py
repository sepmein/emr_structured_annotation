import unittest
from emr_annotation.annotation_analysis.entity_label_errors import residuals, categorize


def entity(start, end, label):
    return dict(start=start,end=end,label=label)


class EntityLabelErrorTests(unittest.TestCase):
    def test_duplicate_correct_prediction_is_not_misclassification(self):
        item=entity(0,2,'甲')
        p,g,common=residuals([item,item],[item])
        self.assertEqual(sum(common.values()),1)
        self.assertEqual(len(g),0)
        self.assertEqual(list(categorize(p,[item],'fp'))[0][1],'duplicate_prediction')

    def test_identical_span_different_label_has_direction(self):
        p,g=entity(0,2,'甲'),entity(0,2,'乙')
        self.assertEqual(list(categorize([p],[g],'fp'))[0][1:],('same_span_wrong_label','乙'))
        self.assertEqual(list(categorize([g],[p],'fn'))[0][1:],('same_span_wrong_label','甲'))

    def test_same_label_boundary_difference_is_separate(self):
        p,g=entity(1,3,'甲'),entity(0,3,'甲')
        self.assertEqual(list(categorize([p],[g],'fp'))[0][1],'same_label_wrong_boundary')
        self.assertEqual(list(categorize([g],[p],'fn'))[0][1],'same_label_wrong_boundary')

    def test_no_overlap_does_not_match_nearby_entity(self):
        p,g=entity(3,5,'甲'),entity(0,3,'甲')
        self.assertEqual(list(categorize([p],[g],'fp'))[0][1],'extra_no_overlap')
        self.assertEqual(list(categorize([g],[p],'fn'))[0][1],'omitted_no_overlap')

    def test_multiple_reference_labels_do_not_invent_unique_target(self):
        rows=list(categorize([entity(0,2,'甲')],[entity(0,2,'乙'),entity(0,2,'丙')],'fp'))
        self.assertEqual(rows[0][1:],('ambiguous_same_span_labels',None))


if __name__=='__main__':unittest.main()
