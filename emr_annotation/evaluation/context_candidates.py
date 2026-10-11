"""Text-only candidate and local-context features; no model or reference lookup."""
import re

PATTERNS={
    '数值':r'\d+(?:\.\d+)?',
    '体温':r'体温|Tmax|T',
    '时间表达':r'(?:19|20)\d{2}[年./-]\d{1,2}[月./-]\d{1,2}(?:日)?|\d+(?:\.\d+)?(?:小时|分钟|个月|天|日|月|年|周)(?:前|后)?|昨晚|昨日|今天|今日|明日|今晨',
}


def candidates(text, entities=(), extra_mentions=None):
    found={(e['start'],e['end'],e['label']) for e in entities if e['label'] in PATTERNS}
    for label,pattern in PATTERNS.items():
        found.update((m.start(),m.end(),label) for m in re.finditer(pattern,text))
    for label,mentions in (extra_mentions or {}).items():
        for mention in mentions:
            found.update((m.start(),m.end(),label) for m in re.finditer(re.escape(mention),text))
    return [dict(start=s,end=e,label=l) for s,e,l in sorted(found)]


def context(text, entity, radius=24, normalize_digits=True):
    s,e=entity['start'],entity['end'];label=entity['label']
    before=text[max(0,s-radius):s];after=text[e:e+radius];mention=text[s:e]
    clean=lambda x:re.sub(r'\d','0',x) if normalize_digits else x
    return clean(before)+'【'+label+'：'+clean(mention)+'】'+clean(after)


def annotation_questions(rows):
    questions=[]
    for row in rows:
        text=row['text']
        for entity in row['entities']:
            s,e=entity['start'],entity['end'];mention=text[s:e]
            if entity['label']=='体温' and mention=='T' and re.match(r'\d{1,2}[-～~]\d{1,2}',text[e:e+10]) and re.search(r'椎|脊髓',text[max(0,s-70):e+70]):
                questions.append(dict(task_id=row['task_id'],start=s,end=e,label=entity['label'],mention=mention,
                                      context=text[max(0,s-70):e+70],status='reference_question_do_not_change_answer',
                                      reason='T followed by vertebral-level range in spine context; needs human/guide recheck'))
    return questions
