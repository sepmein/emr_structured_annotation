import json,re,sys
from pathlib import Path
if len(sys.argv)!=3:
    sys.exit(f"用法: python {Path(__file__).name} <起始行号> <结束行号>")
try:
    start,end=int(sys.argv[1]),int(sys.argv[2])
except ValueError:
    sys.exit("起始行号和结束行号必须是整数")
p=Path(__file__).resolve().parents[1]/"output/emr_data_1_20261008"
a=json.loads((p/"_parsed_records.json").read_text(encoding="utf-8"))
pat=re.compile(r"肺|胸|呼吸|咳|痰|发热|发烧|低热|高热|寒战|畏寒|气促|气急|喘|氧|SpO|PaO|FiO|鼻翼|三凹|休克|衰竭|惊厥|抽搐|拒食|喂养|脱水|流感|新冠|病毒|支原体|衣原体|军团|曲霉|隐球|链球|百日咳|博卡|旅居|疫区|接触|聚集|禽|牲畜|牧民",re.I)
for x in a:
    if not start<=x["row"]<=end: continue
    t=x["data"]["text"]
    print("\nROW",x["row"],"LEN",len(t))
    if len(t)<=650: print(t)
    else:
        chunks=re.split(r"(?<=[。；;\n])",t)
        seen=set()
        for c in chunks:
            c=c.strip()
            if not c or c in seen: continue
            seen.add(c)
            if pat.search(c) or c.startswith(("主诉","诊断依据","确定诊断","出院诊断","初步诊断","现病史","诊疗过程","出院情况")):
                # Long lines are split into comma clauses; retain all matches plus adjacent clause.
                if len(c)>550:
                    clauses=re.split(r"(?<=[，,])",c)
                    idx={j for i,v in enumerate(clauses) if pat.search(v) for j in (i-1,i,i+1) if 0<=j<len(clauses)}
                    print("".join(v if j in idx else "…" for j,v in enumerate(clauses)))
                else: print(c)
