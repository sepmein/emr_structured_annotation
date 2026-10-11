"""Bind explicit textual test evidence to its mention; never infer a diagnosis."""
from copy import deepcopy
import re


def mention_record(text,entity):
    s,e=entity['start'],entity['end']
    line=text[text.rfind('\n',0,s)+1:e]
    structured=bool(re.search(r'(?:检验项目名称|检测项目名称)\s*[:：]',line))
    if structured:
        stop=re.search(r'\n\s*\n\s*操作时间|参考(?:区间|值)',text[e:e+800])
        end=e+stop.start() if stop else min(len(text),e+800)
    else:
        stop=re.search(r'[，,；;。\n]',text[e:e+100])
        end=e+stop.start() if stop else min(len(text),e+100)
    return text[e:end],e


def explicit_attributes(text,entity,schema):
    tail,start=mention_record(text,entity);values={};evidence={}
    f=schema['fields']
    if entity['label'] in f['test_result']['scope']:
        hit=re.search(r'无法判断|可疑|不确定|弱阳性|阳性|阴性',tail)
        if hit:
            # A reference interval alone is not the current test result.
            before=tail[:hit.start()]
            if not re.search(r'参考(?:区间|值)',before):
                value='negative' if hit.group()=='阴性' else ('positive' if '阳性' in hit.group() else 'unknown')
                values['test_result']=value;evidence['test_result']=dict(start=start+hit.start(),end=start+hit.end(),text=hit.group())
    if entity['label'] in f['detection_method']['scope']:
        for pattern,value in [(r'mNGS','mNGS'),(r'(?:IgM|lgM)','抗体IgM'),(r'(?:IgG|lgG)','抗体IgG'),(r'核酸|RNA|DNA|PCR','核酸检测'),(r'抗原','抗原快速检测'),(r'培养','培养')]:
            hit=re.search(pattern,tail,re.I)
            if hit and value in f['detection_method']['options']:
                values['detection_method']=value;evidence['detection_method']=dict(start=start+hit.start(),end=start+hit.end(),text=hit.group());break
    # Specimen evidence must be stated inside this record, not inferred from a neighbouring test.
    if entity['label'] in f['specimen_type']['scope']:
        hit=re.search(r'(?:标本|样本)(?:类型)?\s*[:：]\s*(口咽拭子|鼻咽拭子|深部痰液|痰|肺泡灌洗液|血液|尿)',tail)
        if hit:
            mapping={'口咽拭子':'口咽/鼻咽拭子','鼻咽拭子':'口咽/鼻咽拭子','血液':'血液(未说明采样间隔)'}
            value=mapping.get(hit.group(1),hit.group(1))
            if value in f['specimen_type']['options']:values['specimen_type']=value;evidence['specimen_type']=dict(start=start+hit.start(1),end=start+hit.end(1),text=hit.group(1))
    return values,evidence


def anatomical_conflict(text,entity):
    if entity['label'] not in ('体温','数值'):return False
    s,e=entity['start'],entity['end'];lo=max(0,s-70);hi=min(len(text),e+70)
    if not re.search(r'椎|脊髓',text[lo:hi]):return False
    # Complete level tokens only; a decimal temperature T38.5 is not a level.
    for hit in re.finditer(r'[CTLS](?:1[0-2]|[1-9])(?:\s*[-～~—]\s*[CTLS]?(?:1[0-2]|[1-9]))?(?![\d.])',text[lo:hi]):
        a,b=lo+hit.start(),lo+hit.end()
        if a<=s and e<=b:return True
    return False


def bind(predictions,schema,fields=(),guard=False):
    result=deepcopy(predictions)
    for row in result:
        kept=[]
        for e in row['entities']:
            if guard and anatomical_conflict(row['text'],e):continue
            values,evidence=explicit_attributes(row['text'],e,schema)
            for f,v in values.items():
                if f not in fields:continue
                e.setdefault('attributes',{})[f]=v
                e.setdefault('attribute_evidence',{})[f]=evidence[f]
                e.setdefault('attribute_sources',{})[f]='explicit_text_record_binding'
            kept.append(e)
        row['entities']=kept;ids={e['id'] for e in kept}
        row['relations']=[r for r in row.get('relations',[]) if r['from_id'] in ids and r['to_id'] in ids]
    return result
