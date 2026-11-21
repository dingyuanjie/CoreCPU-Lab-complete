"""Standalone Forward-Forward proof of concept; intentionally NOT wired into A-D.

Layers learn positive vs corrupted input goodness using ONLY layer-local
backpropagation. This is a research *demonstration* of a learning alternative,
not a validated sequence model or a claim of zero local autodiff.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ForwardForwardLayer(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,threshold:float=2.0):
        super().__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        self.threshold=threshold

    def forward(self,x):
        x=x/(x.norm(dim=-1,keepdim=True)+1e-6)
        return F.relu(self.linear(x))

    def goodness(self,x):
        return self.forward(x).square().mean(-1)

    def local_fit(self,positive,negative,optimizer):
        pos=self.goodness(positive)
        neg=self.goodness(negative)
        loss=F.softplus(self.threshold-pos).mean()+F.softplus(neg-self.threshold).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())


def ff_demo(steps:int=30,seed:int=313):
    torch.manual_seed(seed)
    model=ForwardForwardLayer(8,32,threshold=0.15)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    result=[]
    for step in range(steps):
        positive=torch.randn(64,8)*0.2+1
        negative=torch.randn(64,8)*0.2-1
        loss=model.local_fit(positive,negative,optimizer)
        result.append(loss)
    return {'steps':steps,'loss_start':result[0],'loss_last':result[-1],
            'description':'isolated FF demo, not integrated into four architecture comparison'}
