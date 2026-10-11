import unittest
from emr_annotation.evaluation.pathogen_attribute_audit import audit,FIELDS


class PathogenAttributeAuditTests(unittest.TestCase):
    def setUp(self):
        self.schema={'fields':{f:{'scope':['病原甲'],'options':['阳性','阴性','原文未说明']} for f in FIELDS}}

    def entity(self,start,attributes):
        return dict(start=start,end=start+3,label='病原甲',attributes=attributes)

    def test_unannotated_slots_are_excluded_and_missing_entities_are_wrong(self):
        refs=[dict(task_id='sample',entities=[self.entity(0,{'test_result':'阳性'}),self.entity(5,{})])]
        preds=[dict(task_id='sample',entities=[])]
        result=audit(refs,preds,self.schema)
        self.assertEqual(result['fields']['test_result']['known_slots'],1)
        self.assertEqual(result['fields']['test_result']['correct'],0)
        self.assertEqual(result['fields']['specimen_type']['accuracy'],None)
        self.assertEqual(result['joint_known_record']['support'],0)

    def test_repeated_pathogen_is_scored_per_position_and_joint_record(self):
        positive={f:'阳性' for f in FIELDS};negative={f:'阴性' for f in FIELDS}
        refs=[dict(task_id='sample',entities=[self.entity(0,positive),self.entity(5,negative)])]
        preds=[dict(task_id='sample',entities=[self.entity(0,positive),self.entity(5,positive)])]
        result=audit(refs,preds,self.schema)
        self.assertEqual(result['joint_known_record'],dict(support=2,correct=1,accuracy=.5))
        self.assertEqual(result['fields']['test_result']['matched_entities'],2)
        self.assertEqual(result['fields']['test_result']['correct'],1)


if __name__=='__main__':unittest.main()
