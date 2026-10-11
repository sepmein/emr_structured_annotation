"""Token-envelope alignment with explicit character-boundary targets, stdlib only."""


def align_entity(offsets, start, end, max_width, refine=False, max_trim=32):
    real=[(i,s,e) for i,(s,e) in enumerate(offsets) if e>s]
    if refine:
        begins=[i for i,s,e in real if s<=start<e]
        finishes=[i+1 for i,s,e in real if s<end<=e]
    else:
        begins=[i for i,s,e in real if s==start]
        finishes=[i+1 for i,s,e in real if e==end]
    if not begins or not finishes:return None
    begin=min(begins) if refine else max(begins)
    finish=max(finishes)
    if not 0<finish-begin<=max_width:return None
    left=start-offsets[begin][0];right=offsets[finish-1][1]-end
    if not 0<=left<=max_trim or not 0<=right<=max_trim:return None
    return begin,finish,left,right


def suppress_redundant_spans(entities, overlap=.8):
    """Optional confidence-based same-label suppression, preserving other labels.

    Selection is a validation parameter, not a rewrite of reference spans.
    """
    if not overlap:return entities
    kept=[]
    for entity in sorted(entities,key=lambda e:e['score'],reverse=True):
        redundant=False
        for other in kept:
            if entity['label']!=other['label']:continue
            intersection=max(0,min(entity['end'],other['end'])-max(entity['start'],other['start']))
            smaller=min(entity['end']-entity['start'],other['end']-other['start'])
            if smaller and intersection/smaller>=overlap:
                redundant=True;break
        if not redundant:kept.append(entity)
    return sorted(kept,key=lambda e:(e['start'],e['end'],e['label']))
