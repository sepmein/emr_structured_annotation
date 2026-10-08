from pathlib import Path
import re,json
root=Path(__file__).resolve().parents[1]
def change(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new,1)
text=(root/'annotation_agent_workflow/guides/label_dictionary_v2.1.1.md').read_text(encoding='utf-8')
script=(root/'tmp/complete-case-decision-instructions.py').read_text(encoding='utf-8')
exec(script[script.index('# Reuse the four existing definition cards'):])
