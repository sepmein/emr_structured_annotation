from pathlib import Path
import xml.etree.ElementTree as ET
import sys

root=Path(__file__).resolve().parents[1]
sys.path.append(str(root/'.venv/Lib/site-packages'))
from label_studio_tools.core.label_config import parse_config
config=root/'label_studio/pneumonia_config.xml'
before=ET.parse(root/'tmp/pneumonia-before-case-control.xml').getroot()
after=ET.parse(config).getroot()
choices=after.findall('.//Choices')
case=next(c for c in choices if c.get('name')=='case_decision')
expected=['目标病例信号','待专业复核','非目标','信息不足']
assert case.get('choice')=='single-radio'
assert case.get('perRegion')=='false'
assert case.get('required')=='true'
assert case.get('toName')=='chief_complaint_text'
assert [c.get('value') for c in case]==expected
assert not any('selected' in c.attrib or 'checked' in c.attrib for c in case)
parent={child:node for node in after.iter() for child in node}
node=case
while node in parent:
    assert not any(key in node.attrib for key in ['visibleWhen','whenLabelValue','whenChoiceValue'])
    node=parent[node]
names=[n.get('name') for n in after.iter() if n.get('name')]
assert len(names)==len(set(names))
old_controls={n.get('name'):ET.tostring(n) for n in before.iter() if n.get('name')}
new_controls={n.get('name'):ET.tostring(n) for n in after.iter() if n.get('name')}
assert set(new_controls)-set(old_controls)=={'case_decision'}
assert all(new_controls[name]==value for name,value in old_controls.items()), 'An existing control changed'
assert len(after.findall('.//Label'))==49
assert len([c for c in choices if c.get('perRegion')=='true'])==9
assert len(after.findall('.//Relation'))==5
parsed=parse_config(config.read_text(encoding='utf-8'))
assert parsed['case_decision']['type']=='Choices'
assert parsed['case_decision']['labels']==expected
assert parsed['case_decision']['to_name']==['chief_complaint_text']
assert parsed['case_decision']['inputs']==[{'type':'Text','value':'text'}]
print('Label Studio parser passed: case_decision, four choices, task-level, required, no default.')
print('Existing named controls and text template unchanged: 49 entity labels, 9 entity attributes, 5 relations.')
