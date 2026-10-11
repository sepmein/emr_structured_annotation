"""Entity-conditioned field queries over a frozen encoder's local context."""
import math
from torch import nn
import torch


class AttributeContextHead(nn.Module):
    def __init__(self, hidden_size, base_heads, radius=40):
        super().__init__()
        self.radius=radius
        self.key=nn.Linear(hidden_size,128)
        self.value=nn.Linear(hidden_size,128)
        self.queries=nn.ModuleDict({f:nn.Linear(256,128) for f in base_heads})
        self.context_outputs=nn.ModuleDict({f:nn.Linear(128,len(h.weight)) for f,h in base_heads.items()})
        self.base_heads=nn.ModuleDict(base_heads)
        for head in self.context_outputs.values():nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,hidden,representations,spans):
        if not spans:return {f:hidden.new_empty((0,len(head.weight))) for f,head in self.base_heads.items()}
        keys=self.key(hidden.float());values=self.value(hidden.float())
        positions=torch.arange(len(hidden),device=hidden.device).unsqueeze(0)
        starts=torch.tensor([s for s,e,l in spans],device=hidden.device).unsqueeze(1)
        ends=torch.tensor([e for s,e,l in spans],device=hidden.device).unsqueeze(1)
        valid=(positions>=starts-self.radius)&(positions<ends+self.radius)&(positions>0)&(positions<len(hidden)-1)
        outputs={}
        for field,base in self.base_heads.items():
            query=self.queries[field](representations.float())
            attention=(query@keys.T/math.sqrt(128)).masked_fill(~valid,-1e4).softmax(-1)
            outputs[field]=base(representations.float())+self.context_outputs[field](attention@values)
        return outputs
