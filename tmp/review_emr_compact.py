import json,re,sys
from pathlib import Path
a=json.loads(Path('output/emr_data_1_20261008/_parsed_records.json').read_text(encoding='utf-8'))
pat=re.compile(r'肺炎|肺部感染|肺.{0,10}(?:炎症|炎性|感染|渗出|实变|磨玻璃)|发热|发烧|低热|高热|咳嗽|咳痰|气促|气急|喘息|喘鸣|呼吸困难|呼吸衰竭|鼻翼煽动|三凹|休克|器官衰竭|惊厥|抽搐|拒食|喂养困难|脱水征|流感|新冠|冠状病毒|合胞病毒|腺病毒|偏肺|副流感|鼻病毒|肠道病毒|博卡|衣原体|支原体|军团|曲霉|隐球|肺炎链球|百日咳',re.I)
for x in a:
 if not int(sys.argv[1])<=x['row']<=int(sys.argv[2]): continue
 t=x['data']['text']
 print('\n#',x['row'],'chars',len(t))
 if len(t)<650:
  print(re.sub(r'(?:操作时间|诊断时间)：[^\n]+\n','',t))
 else:
  cc=list(dict.fromkeys(re.findall(r'主诉[：:][^\n。]+',t)))
  print('CC:', ' | '.join(cc)[:400])
  his=list(dict.fromkeys(re.findall(r'现病史[：:][^\n]+',t)))
  print('HIS:', ' | '.join(h[:260] for h in his[:2]))
  windows=[]
  labs=[]
  for snippet in re.split(r'[。；\n]',t):
   snippet=snippet.strip()
   if not pat.search(snippet): continue
   if '院内检验项目' in snippet:
    if '名称' in snippet: labs.append(snippet.split('：')[-1])
    continue
   snippet=re.sub(r'^[（(]?\d+[）).、]?','',snippet)
   if len(snippet)>240:
    clauses=re.split(r'[，,]',snippet)
    snippet='，'.join(c for c in clauses if pat.search(c))
   if snippet not in windows: windows.append(snippet)
  print('EVID:', ' | '.join(windows)[:1500])
  if labs: print('LAB:', ','.join(dict.fromkeys(labs))[:450])
  if not windows and not his and not cc: print(t[:800])
