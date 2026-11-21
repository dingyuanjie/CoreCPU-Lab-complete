"""Structured model with a FACTORIZED organization and compute mechanism."""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from corecpu_lab.core.config import ModelConfig
from corecpu_lab.core.types import ModelOutput,detach_state
from corecpu_lab.organizations.runtime import build_runtime
from corecpu_lab.tasks.structured.dataset import Q,N_KEYS

class ExternalReader(nn.Module):
    """Same address-matching inductive bias for every factorial combination."""
    def __init__(self,key_dim:int=24):
        super().__init__()
        self.keys=nn.Embedding(N_KEYS,key_dim)
        self.log_scale=nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self,keys:Tensor,values:Tensor,queries:Tensor):
        k=F.normalize(self.keys(keys).float(),dim=-1)
        q=F.normalize(self.keys(queries).float(),dim=-1)
        logits=q@k.transpose(1,2)*self.log_scale.exp().clamp(max=80)
        value_dist=F.one_hot(values,Q).float()
        return logits.softmax(-1)@value_dist, logits

class StructuredModel(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        cfg.validate()
        if cfg.task!='structured':raise ValueError('StructuredModel requires task=structured')
        self.config=cfg
        d=cfg.hidden_dim
        self.reader=ExternalReader()
        self.op=nn.Embedding(4,d)
        self.project=nn.Sequential(nn.Linear(2*Q+1,d),nn.GELU(),nn.LayerNorm(d))
        self.runtime=build_runtime(cfg.organization,cfg.compute,d,cfg.ff_dim,cfg.depth,
                                   cfg.num_slots,cfg.num_experts,cfg.top_k)
        self.head=nn.Linear(d,Q)

    def forward(self,keys:Tensor,values:Tensor,queries:Tensor,use_ref:Tensor,
                literals:Tensor,ops:Tensor,mod:Tensor, *, local_detach:bool=False)->ModelOutput:
        read,attention=self.reader(keys,values,queries)
        state=read[:,0]
        arguments=torch.where(use_ref[...,None],read[:,1:],F.one_hot(literals,Q).float())
        cap=mod.float()[:,None]/Q
        invalid=torch.arange(Q,device=mod.device)[None,:]>=mod[:,None]
        rt_state=None
        outputs,features,all_routes=[],[],[]
        total_balance=[]
        counts=[0]*self.config.num_experts if self.config.organization=='corecpu' else []
        for t in range(ops.shape[1]):
            if local_detach:
                state=state.detach()
                rt_state=detach_state(rt_state)
            input_x=self.project(torch.cat((state,arguments[:,t],cap),dim=-1))+self.op(ops[:,t])
            hidden,rt_state,balance,c,route=self.runtime.step(input_x,rt_state)
            logits=self.head(hidden).float().masked_fill(invalid,-1e4)
            state=logits.softmax(-1)  # free-running differentiable state; NO true intermediate labels
            outputs.append(logits)
            features.append(hidden)
            total_balance.append(balance)
            if route is not None:all_routes.append(route.mean(1))
            if counts: counts=[a+b for a,b in zip(counts,c)]
        return ModelOutput(
            logits=torch.stack(outputs,1),
            features=torch.stack(features,1),
            attention_logits=attention,
            route_logits=torch.stack(all_routes,1) if all_routes else None,
            auxiliary_loss=torch.stack(total_balance).mean(),
            expert_counts=counts,
            state=rt_state,
        )
