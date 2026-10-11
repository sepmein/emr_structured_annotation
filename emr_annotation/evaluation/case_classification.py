"""Existing case metric contract shared by training and offline reports."""
from collections import Counter

def case_metrics(target,predicted,options):
    per=[]
    for i,name in enumerate(options):
        tp=sum(a==i and b==i for a,b in zip(target,predicted));fp=sum(a!=i and b==i for a,b in zip(target,predicted));fn=sum(a==i and b!=i for a,b in zip(target,predicted))
        per.append(dict(label=name,support=tp+fn,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    supported=[p['f1'] or 0. for p in per if p['support']]
    return dict(support=len(target),accuracy=sum(a==b for a,b in zip(target,predicted))/len(target) if target else None,
                macro_f1=sum(supported)/len(supported) if supported else None,per_class=per,
                confusion=[dict(reference=options[a],prediction=options[b],count=n) for (a,b),n in Counter(zip(target,predicted)).items()])
