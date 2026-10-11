"""Learned late fusion of mmBERT text and predicted structured evidence."""
import torch
from torch import nn
from torch.nn import functional as F


class Heads(nn.Module):
    def __init__(self, size, cases, syndromes, pathogens, hierarchical=False):
        super().__init__()
        self.hierarchical=hierarchical
        self.case = nn.Linear(size+syndromes+pathogens if hierarchical else size,3 if hierarchical else cases)
        self.syndrome = nn.Sequential(nn.Linear(size,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,syndromes)) if hierarchical else nn.Linear(size,syndromes)
        self.pathogen = nn.Sequential(nn.Linear(size,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,pathogens)) if hierarchical else nn.Linear(size,pathogens)

    def forward(self, hidden):
        s,p=self.syndrome(hidden),self.pathogen(hidden)
        if self.hierarchical:
            info,signal,certainty=self.case(torch.cat([hidden,s.sigmoid(),p.sigmoid()],-1)).unbind(-1)
            c=torch.stack([F.logsigmoid(-info)+F.logsigmoid(signal)+F.logsigmoid(certainty),
                           F.logsigmoid(-info)+F.logsigmoid(signal)+F.logsigmoid(-certainty),
                           F.logsigmoid(-info)+F.logsigmoid(-signal),F.logsigmoid(info)],-1)
        else:c=self.case(hidden)
        return dict(case=c,syndrome=s,pathogen=p)


class EvidenceFusionModel(nn.Module):
    def __init__(self, text_size, evidence_size, cases=4, syndromes=4, pathogens=4, hidden_size=128, hierarchical_case=False):
        super().__init__()
        self.text_projection=nn.Sequential(nn.Linear(text_size,hidden_size),nn.LayerNorm(hidden_size),nn.GELU(),nn.Dropout(.25))
        self.evidence_projection=nn.Sequential(nn.Linear(evidence_size,hidden_size),nn.LayerNorm(hidden_size),nn.GELU(),nn.Dropout(.25))
        self.text_heads=Heads(hidden_size,cases,syndromes,pathogens,hierarchical_case)
        self.evidence_heads=Heads(hidden_size,cases,syndromes,pathogens,hierarchical_case)
        self.gate=nn.Sequential(nn.Linear(hidden_size*2,hidden_size),nn.Sigmoid())
        self.fusion_projection=nn.Sequential(nn.Linear(hidden_size*2+cases*2,hidden_size),nn.GELU(),nn.Dropout(.2))
        self.fusion_heads=Heads(hidden_size,cases,syndromes,pathogens,hierarchical_case)

    def forward(self, text, evidence):
        t=self.text_projection(text);e=self.evidence_projection(evidence)
        text_result=self.text_heads(t);evidence_result=self.evidence_heads(e)
        gate=self.gate(torch.cat([t,e],-1))
        fused=self.fusion_projection(torch.cat([gate*t,(1-gate)*e,text_result['case'].softmax(-1),evidence_result['case'].softmax(-1)],-1))
        return dict(text=text_result,evidence=evidence_result,fusion=self.fusion_heads(fused),gate=gate)
