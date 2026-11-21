"""A/B: causal attention + dense/shared MoE. C/D: attention-free causal SEC.

Input is UTF-8 bytes (0..255), optionally BOS=256. No future token can
influence the prediction of an earlier token.
"""
from __future__ import annotations
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from corecpu_lab.core.config import ModelConfig
from corecpu_lab.core.types import ModelOutput
from corecpu_lab.compute.attention import CausalAttention,SparseExperts
from corecpu_lab.organizations.runtime import build_runtime
from corecpu_lab.tasks.language.dataset import VOCAB,BOS

class DenseLMBlock(nn.Module):
    def __init__(self,d:int,hidden:int,heads:int):
        super().__init__()
        self.a_norm=nn.LayerNorm(d)
        self.f_norm=nn.LayerNorm(d)
        self.attn=CausalAttention(d,heads)
        self.ff=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
    def forward(self,x:Tensor):
        x=x+self.attn(self.a_norm(x))
        return x+self.ff(self.f_norm(x))

class SharedLMBlock(nn.Module):
    def __init__(self,d:int,hidden:int,heads:int,experts:int,top_k:int):
        super().__init__()
        self.a_norm=nn.LayerNorm(d)
        self.f_norm=nn.LayerNorm(d)
        self.attn=CausalAttention(d,heads)
        self.ff=SparseExperts(d,hidden,experts,top_k)
    def forward(self,x:Tensor):
        x=x+self.attn(self.a_norm(x))
        y,balance,counts=self.ff(self.f_norm(x))
        return x+y,balance,counts

class ByteLanguageModel(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        cfg.validate()
        if cfg.task!='language':raise ValueError('ByteLanguageModel needs task=language')
        self.config=cfg
        d=cfg.hidden_dim
        self.token=nn.Embedding(VOCAB,d)
        self.final_norm=nn.LayerNorm(d)
        self.position=nn.Embedding(cfg.context,d) if cfg.compute=='conventional' else None
        self.depth_embed=nn.Embedding(cfg.depth,d) if (cfg.compute=='conventional' and cfg.organization=='corecpu') else None
        if cfg.compute=='conventional':
            if cfg.organization=='sequential':
                self.blocks=nn.ModuleList([DenseLMBlock(d,cfg.ff_dim,cfg.heads) for _ in range(cfg.depth)])
            else:
                self.shared_block=SharedLMBlock(d,cfg.ff_dim,cfg.heads,cfg.num_experts,cfg.top_k)
            self.runtime=None
        else:
            self.runtime=build_runtime(cfg.organization,'sec',d,cfg.ff_dim,cfg.depth,
                                       cfg.num_slots,cfg.num_experts,cfg.top_k)

    def forward(self,tokens:Tensor, *, state=None)->ModelOutput:
        if tokens.ndim!=2:raise ValueError('tokens must be [B,T]')
        b,t=tokens.shape
        if t<1 or t>self.config.context:raise ValueError('sequence must fit model context')
        x=self.token(tokens)
        balances=[]
        counts=[0]*self.config.num_experts if self.config.organization=='corecpu' else []
        if self.config.compute=='conventional':
            if state is not None:raise ValueError('attention baselines do not support external state')
            x=x+self.position(torch.arange(t,device=tokens.device))[None]
            if self.config.organization=='sequential':
                for block in self.blocks:
                    x=block(x)
            else:
                for dep in range(self.config.depth):
                    x,balance,c=self.shared_block(x+self.depth_embed.weight[dep])
                    balances.append(balance)
                    counts=[a+v for a,v in zip(counts,c)]
        else:
            # A causal streaming recurrence. No positional embedding is added:
            # the history is already represented by the carry state.
            rows=[]
            for pos in range(t):
                h,state,balance,c,_=self.runtime.step(x[:,pos],state)
                rows.append(h)
                balances.append(balance)
                if counts:counts=[a+v for a,v in zip(counts,c)]
            x=torch.stack(rows,dim=1)
        features=self.final_norm(x)
        logits=F.linear(features,self.token.weight)
        balance=torch.stack(balances).mean() if balances else logits.new_zeros(())
        return ModelOutput(logits=logits,auxiliary_loss=balance,features=features,
                           state=state,expert_counts=counts)

    @torch.no_grad()
    def generate(self,prompt:str='',new_tokens:int=80,temperature:float=0.8,top_k:int=40)->str:
        if new_tokens<0:raise ValueError('new_tokens must be >= 0')
        if temperature<0:raise ValueError('temperature must be >= 0')
        prefix=list(prompt.encode('utf-8')) or [BOS]
        sequence=list(prefix)
        device=self.token.weight.device
        was_training=self.training
        self.eval()
        try:
            for _ in range(new_tokens):
                input_tokens=torch.tensor([sequence[-self.config.context:]],device=device)
                scores=self(input_tokens).logits[0,-1,:256].float()
                if temperature==0:
                    new=int(scores.argmax().item())
                else:
                    scores=scores/max(temperature,1e-6)
                    if top_k>0 and top_k<256:
                        values,inds=torch.topk(scores,top_k)
                        mask=torch.full_like(scores,-1e9)
                        scores=mask.scatter(0,inds,values)
                    new=int(torch.multinomial(scores.softmax(-1),1).item())
                sequence.append(new)
        finally:
            self.train(was_training)
        return bytes(sequence[1:] if prefix==[BOS] else sequence).decode('utf-8',errors='replace')
