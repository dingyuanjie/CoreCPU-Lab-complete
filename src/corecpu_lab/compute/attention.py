"""Causal attention and actual sparse FFN dispatch for language baselines."""
from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self,dim:int,heads:int):
        super().__init__()
        if dim%heads:raise ValueError('dim must be divisible by heads')
        self.heads=heads
        self.qkv=nn.Linear(dim,3*dim,bias=False)
        self.out=nn.Linear(dim,dim,bias=False)

    def forward(self,x:Tensor)->Tensor:
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).contiguous().reshape(b,t,d))

class SparseExperts(nn.Module):
    def __init__(self,dim:int,hidden:int,experts:int,top_k:int):
        super().__init__()
        if not 1<=top_k<=experts:raise ValueError('bad experts/top_k')
        self.num_experts,self.top_k=experts,top_k
        self.router=nn.Linear(dim,experts)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim)) for _ in range(experts)])

    def forward(self,x:Tensor)->tuple[Tensor,Tensor,list[int]]:
        shape=x.shape
        flat=x.reshape(-1,shape[-1])
        logits=self.router(flat)
        probs=logits.softmax(-1)
        chosen_p,chosen=probs.topk(self.top_k,dim=-1)
        if self.top_k==1:
            # Forward = 1; backward follows selected softmax probability.
            weights=1+chosen_p-chosen_p.detach()
        else:
            weights=chosen_p/chosen_p.sum(-1,keepdim=True).clamp_min(1e-8)
        out=torch.zeros_like(flat)
        counts=[]
        for eid,expert in enumerate(self.experts):
            token,rank=(chosen==eid).nonzero(as_tuple=True)
            counts.append(int(token.numel()))
            if token.numel():
                y=expert(flat.index_select(0,token))*weights[token,rank,None]
                out=out.index_add(0,token,y)
        fraction=torch.tensor(counts,device=flat.device,dtype=probs.dtype)/max(1,sum(counts))
        balance=self.num_experts*(probs.mean(0)*fraction).sum()
        return out.reshape(shape),balance,counts
