"""Freeze human-reviewed cases and construct traceable rule-labelled augmentation.

Generated records are explicitly synthetic and automatically checked; they are
never relabelled as human-reviewed or expert gold. Real patient splits stay fixed.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.adjudication.ai_sample import validate_samples
from emr_annotation.training_data.schema_multitask import sha256, write_json
from emr_annotation.evaluation.clinical_evidence import extract, contract


class Builder:
    def __init__(self, schema):
        self.schema, self.text, self.entities, self.relations = schema, '', [], []

    def add(self, text, label=None, **attributes):
        start = len(self.text)
        self.text += text
        if label is None:
            return None
        defaults = dict(finding_context='known_present', temporality='current',
                        conditional='no', hypothetical='no', experiencer='patient')
        if label in self.schema['fields']['test_result']['scope']:
            defaults = dict(test_result='unknown', specimen_type='原文未说明', detection_method='原文未说明')
        defaults.update(attributes)
        attrs = {f:v for f,v in defaults.items() if label in self.schema['fields'][f]['scope']}
        e = dict(id=f'e{len(self.entities)}',start=start,end=len(self.text),text=text,label=label,attributes=attrs)
        self.entities.append(e)
        return e['id']

    def edge(self, a, b, kind):
        self.relations.append(dict(from_id=a,to_id=b,type=kind,direction='right',reason='合成模板明示语义归属',rule_ref='synthetic_template_v1'))


def generate(schema, count=1000, seed=42):
    if count < 49:
        raise ValueError('At least 49 records required for the coverage target')
    rng = random.Random(seed)
    pathogens = [l for c in ('pathogen','bacteria','other_pathogen') for l in schema['controls'][c]]
    extra = [*schema['controls']['symptons_labels'],*schema['controls']['epidemics_labels']]
    exposure = {'周围聚集性发病现象':'同住的多人相继发热', '高风险职业史':'从事禽类养殖工作',
                '可疑动物或动物制品接触史':'接触病死禽类', '可疑环境暴露史':'接触被鸟粪污染的环境',
                '境内外旅居史':'近期曾外出旅行'}
    records = []
    for i in range(count):
        b = Builder(schema)
        scenario = i % 12
        family = ['病程叙述','分段记录','复诊摘要'][i%3]
        b.add(f'{family}，记录编号S{rng.randrange(100000,999999)}：')
        if scenario in (3,7,11):
            b.add(rng.choice(['病史同前，检查如前，关键附件未提供。','正文截断，缺少主诉和检查内容。','病种与时间信息缺失，待回源补齐。']))
            decision = '信息不足'
        else:
            if scenario in (0,4,8):
                b.add('本次胸部CT明确提示')
                pneumonia=b.add(rng.choice(['右肺炎症','两肺炎症','肺实变']),'肺炎影像',**({'clinical_course':'improving'} if scenario==8 else {}))
                b.add('，临床诊断');b.add('肺炎','肺炎诊断');b.add('。')
                decision = '目标病例信号'
            elif scenario in (1,5,9):
                b.add('本次考虑');pneumonia=b.add('肺炎','肺炎诊断',finding_context='possible');b.add('可能，尚待影像确认。')
                decision = '待专业复核'
            elif scenario==2:
                b.add('患者否认');pneumonia=b.add('肺炎','肺炎诊断',finding_context='known_absent');b.add('，本次因皮疹复诊。')
                decision = '非目标'
            elif scenario==6:
                b.add('既往曾患');pneumonia=b.add('肺炎','肺炎诊断',temporality='historical',clinical_course='resolved');b.add('，已痊愈，本次无呼吸主诉。')
                decision = '非目标'
            else:
                mode=(i//12)%3
                if mode==0:
                    b.add('患者家属曾患');pneumonia=b.add('肺炎','肺炎诊断',experiencer='family',temporality='historical');b.add('，患者本人本次因皮疹就诊。')
                elif mode==1:
                    b.add('若未来出现呼吸症状，再考虑检查以排查');pneumonia=b.add('肺炎','肺炎诊断',finding_context='possible',temporality='future',hypothetical='yes',conditional='yes');b.add('，当前仅因皮疹就诊。')
                else:
                    b.add('拟行影像检查以排查');pneumonia=b.add('肺炎','肺炎诊断',finding_context='possible',temporality='future',hypothetical='yes');b.add('，尚未检查，本次因皮疹就诊。')
                decision = '非目标'
            # Symptoms are negated in non-target controls so ancillary additions
            # cannot contradict the explicitly constructed case decision.
            negative = decision=='非目标'
            if not negative:
                days=rng.randint(1,9)
                b.add('患者在本次住院中出现');fever=b.add('发热','发热',**({'clinical_course':'resolved'} if i%4==0 else {}));b.add('持续')
                duration=b.add(f'{days}天','时间表达');b.edge(fever,duration,'持续时长')
                b.add('，该症状起始于');begin=b.add(f'{days}天前','时间表达');b.edge(fever,begin,'起始时间')
                if i%4==0:
                    b.add('，已退热，该症状结束于');end=b.add('今日','时间表达');b.edge(fever,end,'结束时间')
                b.add('。')
            for j in range(2):
                label=extra[(i+j)%len(extra)]
                b.add('否认' if negative else '本次有')
                value=exposure.get(label,label)
                b.add(value,label,finding_context='known_absent' if negative else 'known_present');b.add('。')
            indicator=schema['controls']['measure_entities'][i%5]
            number={'体温':rng.choice(['37.2','38.1','39.0']), '呼吸频率':str(rng.choice([18,24,30,42])),
                    '静息状态指氧饱和度':str(rng.choice([89,93,97])), '动脉氧分压(PaO2)':str(rng.choice([55,75,90])),
                    '吸氧浓度(FiO2)':str(rng.choice([21,35,60]))}[indicator]
            unit={'体温':'℃','呼吸频率':'次/分','静息状态指氧饱和度':'%', '动脉氧分压(PaO2)':'mmHg','吸氧浓度(FiO2)':'%'}[indicator]
            b.add('既往测量' if negative else '本次测量')
            a=b.add(indicator,indicator,temporality='historical' if negative else 'current');b.add('为')
            if i%4==0:
                comp=b.add(rng.choice(['≥','≤']),'比较符');b.edge(a,comp,'测量')
            v=b.add(number,'数值');u=b.add(unit,'单位');b.edge(a,v,'测量');b.edge(a,u,'测量');b.add('。')
            label=pathogens[(i//2)%len(pathogens)]
            result=['positive','negative','unknown'][i%3]
            b.add('本次病原检测：');lab=b.add(label,label,test_result=result,specimen_type='痰',detection_method='核酸检测')
            b.add({'positive':'检出','negative':'未检出','unknown':'结果待回报'}[result]);b.add('。')
            b.add('病原记录日期');date=b.add(f'2026-{rng.randint(1,9):02d}-{rng.randint(1,28):02d}','时间表达');b.edge(lab,date,'发生时间');b.add('。')
        text_hash=hashlib.sha256(b.text.encode()).hexdigest()
        records.append(dict(task_id=f'synthetic_{seed}_{i:04d}',patient_id=f'synthetic_patient_{seed}_{i:04d}',project=-1,source_task_id=i,
                            text=b.text,text_sha256=text_hash,entities=b.entities,relations=b.relations,case_decision=decision,
                            case_choices={'case_decision':decision},case_reason=f'构造场景{scenario}，指南3.2/4.1的明示条件；程序校验实体及关系，不是新增人工审查',
                            case_rule='synthetic_explicit_context_v1',ner_supervised=True,relations_supervised=True,relations_complete=True,
                            reference_status='synthetic_rule_adjudicated_not_human_gold',expert_signed=False,synthetic=True,
                            provenance={'generator':'human_reviewed.generate_v2','seed':seed,'scenario':scenario,'template_family':family,'source_patient':None},issues=[]))
    validate_samples(records,schema)
    if len({r['text'] for r in records}) != count:
        raise ValueError('Duplicate synthetic texts')
    return records


def prepare(reference_path,schema_path,preserved_split_dir,output_dir,count=1000,seed=42):
    source=Path(reference_path);schema=load_double_annotation_schema(schema_path)
    records=json.loads(source.read_text('utf-8'))
    validate_samples(records,schema)
    if any(r.get('reference_status')!='human_reviewed_not_expert_gold' or not r.get('human_reviewer','').strip() for r in records):
        raise ValueError('Only frozen human-reviewed reference records accepted')
    assignment,previous,hashes={},{},{}
    for name in ('train','validation','test'):
        p=Path(preserved_split_dir)/(name+'.json');hashes[str(p.resolve())]=sha256(p)
        for r in json.loads(p.read_text('utf-8')):
            key=(r['project'],r['source_task_id'])
            if key in assignment and assignment[key]!=name:raise ValueError('Prior case split leakage')
            assignment[key]=name;previous[key]=r
    if set(assignment)!={(r['project'],r['source_task_id']) for r in records}:raise ValueError('Changed real-case scope')
    splits={n:[] for n in ('train','validation','test')};owners={};texts={}
    for raw in records:
        r=deepcopy(raw);key=(r['project'],r['source_task_id']);old=previous[key];name=assignment[key]
        if r['text']!=old['text'] or r['patient_id']!=old['patient_id']:raise ValueError('Changed source patient/text')
        if owners.setdefault(r['patient_id'],name)!=name or texts.setdefault(r['text'],name)!=name:raise ValueError('Real patient/text leakage')
        r.update(case_choices={'case_decision':r['case_decision']},ner_supervised=True,relations_supervised=True,relations_complete=False,synthetic=False)
        splits[name].append(r)
    synthetic=generate(schema,count,seed)
    # All synthetic augmentation is confined to training. The real validation
    # and test sets stay byte-for-byte equivalent to frozen human references.
    splits['train'].extend(synthetic)
    for rows in splits.values():
        for r in rows:r['evidence_targets']=extract(r['text'],r['entities'],r['relations'],schema)
    coverage={label:dict(real=sum(e['label']==label for r in records for e in r['entities']),synthetic=sum(e['label']==label for r in synthetic for e in r['entities'])) for label in schema['labels']}
    missing=[k for k,v in coverage.items() if not v['synthetic']]
    if missing:raise ValueError('Synthetic missing label coverage: '+str(missing))
    if set(schema['relations'])!={x['type'] for r in synthetic for x in r['relations']}:raise ValueError('Synthetic relation coverage incomplete')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    for name,rows in splits.items():write_json(out/(name+'.json'),rows)
    write_json(out/'reference.json',records);write_json(out/'synthetic_1000.json',synthetic)
    ledger=[]
    for row in synthetic:
        for e in row['entities']:
            ledger.append(dict(task_id=row['task_id'],kind='entity',entity_id=e['id'],decision='constructed_and_schema_validated',label=e['label'],start=e['start'],end=e['end'],attributes=e['attributes']))
        for rel in row['relations']:ledger.append(dict(task_id=row['task_id'],kind='relation',decision='explicit_template_semantic_edge',**rel))
        ledger.append(dict(task_id=row['task_id'],kind='case',decision=row['case_decision'],reason=row['case_reason'],human_reviewed=False,expert_signed=False))
    write_json(out/'synthetic_adjudication_ledger.json',ledger)
    train_schema={'labels':schema['labels'],'controls':schema['controls'],'relations':schema['relations'],
                  'fields':{k:{**v,'options':list(v['values'])} for k,v in schema['fields'].items()},'schema_sha256':sha256(schema_path)}
    write_json(out/'schema.json',train_schema);(out/'label_config.xml').write_bytes(Path(schema_path).read_bytes())
    write_json(out/'evidence_contract.json',contract(schema))
    manifest=dict(stage='training_data',run_id=out.name,created_at=datetime.now(timezone.utc).isoformat(),reference_status='human_reviewed_plus_explicit_synthetic_rules',
                  human_cases=len(records),synthetic_cases=len(synthetic),expert_signed=False,source=str(source.resolve()),source_sha256=sha256(source),
                  preserved_split_files=hashes,split_counts={n:len(rows) for n,rows in splits.items()},
                  real_split_counts={n:sum(not r['synthetic'] for r in rows) for n,rows in splits.items()},
                  verified_no_patient_overlap=True,verified_no_exact_text_overlap=True,synthetic_training_only=True,
                  auxiliary_targets='Rule-derived evidence groups; not separately human-adjudicated syndrome/etiology',label_coverage=coverage,
                  synthetic_cases_by_decision=dict(Counter(r['case_decision'] for r in synthetic)),synthetic_relations=dict(Counter(x['type'] for r in synthetic for x in r['relations'])),
                  near_duplicate_check='Template variants intentionally share semantics; not independent evaluation. Real near-duplicate audit separate.',
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree=subprocess.check_output(['git','status','--porcelain'],text=True))
    write_json(out/'split_manifest.json',manifest);write_json(out/'preparation_audit.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('reference','schema','preserve_splits_from','output_dir'):p.add_argument('--'+field.replace('_','-'),type=Path,required=True)
    p.add_argument('--count',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print(prepare(a.reference,a.schema,a.preserve_splits_from,a.output_dir,a.count,a.seed)['split_counts'])


if __name__=='__main__':main()
