"""Conventional and candidate State Evolution Cell (SEC).

SEC is a *hypothesis*, not a proven novel mechanism; it is deliberately small,
explicitly gated, strictly causal and stateful. No self-attention over past tokens.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ConventionalCell(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fn=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim))

    def forward(self,x:Tensor,state:Tensor|None=None):
        return self.fn(x), None

class StateEvolutionCell(nn.Module):
    """S slots x D features, with feature-wise gated recurrence and slot mixing.

    Features generated from the previous state; stable initialization uses
    negative update gate bias and mild residual. Implementation is sequential
    over tokens and can be significantly slower than optimized attention.
    """
    def __init__(self,dim:int,num_slots:int):
        super().__init__()
        self.dim,self.num_slots=dim,num_slots
        self.in_proj=nn.Linear(dim,dim)
        self.gate=nn.Linear(2*dim,num_slots*dim)
        self.candidate=nn.Linear(dim,num_slots*dim)
        self.decay=nn.Parameter(torch.zeros(num_slots,dim))
        self.out=nn.Linear(dim,dim)
        self.norm=nn.LayerNorm(dim)
        nn.init.constant_(self.gate.bias,-1.0)
        self._local_records: list[tuple[Tensor,Tensor]]=[]

    def initial_state(self,batch:int,device:torch.device,dtype:torch.dtype) -> Tensor:
        return torch.zeros(batch,self.num_slots,self.dim,device=device,dtype=dtype)

    def forward(self,x:Tensor,state:Tensor|None=None):
        if state is None:
            state=self.initial_state(x.size(0),x.device,x.dtype)
        if state.shape!=(x.size(0),self.num_slots,self.dim):
            raise ValueError('SEC state must be [B,num_slots,dim]')
        inp=self.in_proj(x)
        pooled=state.mean(1)
        gate=torch.sigmoid(self.gate(torch.cat((inp,pooled),-1)))\
            .view(x.size(0),self.num_slots,self.dim)
        c=self.candidate(inp).view(x.size(0),self.num_slots,self.dim)
        candidate=torch.tanh(c + state * torch.tanh(self.decay))
        next_state=(1-gate)*state+gate*candidate
        pooled_new=self.norm(next_state.mean(1))
        delta=self.out(pooled_new)
        # Caches are intentionally disabled in BP and enabled only for manual
        # three-factor experiments; clearing after the batch avoids growth.
        if self._collect_local:
            self._local_records.append((pooled_new.detach(),torch.tanh(delta.detach())))
        return delta,next_state

    _collect_local=False

    def begin_local(self):
        self._local_records.clear()
        self._collect_local=True

    @torch.no_grad()
    def apply_local_feedback(self,feedback:Tensor,lr:float,decay:float=0.0001):
        """Exploratory scalar three-factor update to SEC output projection ONLY.

        This is not BPTT nor an exact e-prop implementation. Remaining SEC
        gates stay fixed in ELIG mode. Feedback is per-example, centered.
        """
        if not self._local_records:
            self._collect_local=False
            return
        for features,act in self._local_records:
            if features.shape[0]!=feedback.shape[0]:
                # Component dispatch can have a routed sub-batch. In that case
                # all available rows receive aggregate scalar feedback.
                r=feedback.mean().expand(features.shape[0])
            else:
                r=feedback
            eligibility=torch.einsum('b,bo,bi->oi',r,act,features)/max(1,features.shape[0])
            self.out.weight.add_(lr*eligibility.clamp(-0.1,0.1))
            self.out.weight.mul_(1-lr*decay)
        self._local_records.clear()
        self._collect_local=False


def build_cell(kind:str,dim:int,hidden:int,slots:int)->nn.Module:
    if kind=='conventional':return ConventionalCell(dim,hidden)
    if kind=='sec':return StateEvolutionCell(dim,slots)
    raise ValueError(kind)
