import json,re,sys
from pathlib import Path
a=json.loads(Path('output/emr_data_1_20261008/_parsed_records.json').read_text(encoding='utf-8'))
pat=re.compile(r'肺炎|肺部感染|肺.{0,10}(?:炎症|炎性|感染|渗出|实变|磨玻璃)|发热|发烧|低热|高热|咳嗽|咳痰|气促|气急|喘息|喘鸣|呼吸困难|呼吸衰竭|鼻翼煽动|三凹|休克|器官衰竭|惊厥|抽搐|拒食|喂养困难|脱水征|流感|新冠|冠状病毒|合胞病毒|腺病毒|偏肺|副流感|鼻病毒|肠道病毒|博卡|衣原体|支原体|军团|曲霉|隐球|百日咳',re.I)
for x in a:
 if not int(sys.argv[1])<=x['row']<=int(sys.argv[2]): continue
 t=x['data']['text']
 cc=list(dict.fromkeys(re.findall(r'主诉[：:][^\n。]+',t)))
 his=list(dict.fromkeys(re.findall(r'现病史[：:][^\n]+',t)))
 if len(t)<650:
  t=re.sub(r'(?:操作时间|诊断时间)：[^\n]+\n','',t)
  t=re.sub(r'【[^】]+】','',t)
  t=re.sub(r'(?:院内检验项目代码|院内检验定性结果代码|标准化检验定性结果代码|检验定量结果超出或低于参考值)：[^\n]+','',t)
  print(x['row'],re.sub(r'\n+',' | ',t).strip(' |'))
 else:
  ev=[]
  for s in re.split(r'[。；\n]',t):
   if not pat.search(s): continue
   if len(s)>150: s='，'.join(c for c in re.split(r'[，,]',s) if pat.search(c))
   s=s.strip()
   if s not in ev: ev.append(s)
  print(x['row'],'CC:', ';'.join(cc)[:100],'HIS:', ';'.join(his)[:180],'EX:', ';'.join(ev)[:480])
