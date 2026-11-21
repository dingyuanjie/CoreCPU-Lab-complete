"""Factorial organization implementations; these do not know task labels."""
from __future__ import annotations
import torch
from torch import nn, Tensor
from corecpu_lab.compute.cells import build_cell

class SequentialRuntime(nn.Module):
    """Depth-wise *independent* computation cells; no dynamic expert router."""
    def __init__(self,kind:str,dim:int,hidden:int,depth:int,slots:int):
        super().__init__()
        self.cells=nn.ModuleList([build_cell(kind,dim,hidden,slots) for _ in range(depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.kind=kind

    def initial_state(self,batch:int,device:torch.device,dtype:torch.dtype):
        return [None if self.kind=='conventional' else cell.initial_state(batch,device,dtype) for cell in self.cells]

    def step(self,x:Tensor,state:list|None=None):
        if state is None:state=self.initial_state(x.size(0),x.device,x.dtype)
        new_state=[]
        for cell,norm,old in zip(self.cells,self.norms,state):
            delta,updated=cell(x,old)
            x=norm(x+delta)
            new_state.append(updated)
        return x,new_state,x.new_zeros(()),[],None

class CoreCPURuntime(nn.Module):
    """Shared sparse bank across repeated steps/depth, with true selected execution.

    Gradients can flow through selected expert and its router probabilities;
    experts not selected are NOT executed. All experts currently reside in RAM/VRAM.
    """
    def __init__(self,kind:str,dim:int,hidden:int,depth:int,slots:int,experts:int,top_k:int):
        super().__init__()
        if top_k>experts:raise ValueError('top_k exceeds experts')
        self.kind,self.depth,self.top_k=kind,depth,top_k
        self.experts=nn.ModuleList([build_cell(kind,dim,hidden,slots) for _ in range(experts)])
        self.router=nn.Linear(dim,experts)
        self.norm=nn.LayerNorm(dim)

    def initial_state(self,batch:int,device:torch.device,dtype:torch.dtype):
        return [None if self.kind=='conventional' else cell.initial_state(batch,device,dtype) for cell in self.experts]

    def step(self,x:Tensor,state:list|None=None):
        if state is None:state=self.initial_state(x.size(0),x.device,x.dtype)
        count=[0]*len(self.experts)
        routing=[]
        aux=[]
        for _ in range(self.depth):
            logits=self.router(x)
            probs=logits.softmax(-1)
            chosen_p,chosen=probs.topk(self.top_k,dim=-1)
            if self.top_k==1:
                weights=1+chosen_p-chosen_p.detach()
            else:
                weights=chosen_p/chosen_p.sum(-1,keepdim=True).clamp_min(1e-8)
            delta=torch.zeros_like(x)
            next_states=list(state)
            per_step=[]
            for eid,cell in enumerate(self.experts):
                token,rank=(chosen==eid).nonzero(as_tuple=True)
                num=int(token.numel())
                count[eid]+=num
                per_step.append(num)
                if num==0:continue
                selected_x=x.index_select(0,token)
                selected_state=state[eid].index_select(0,token) if state[eid] is not None else None
                update,selected_new=cell(selected_x,selected_state)
                delta=delta.index_add(0,token,update*weights[token,rank,None])
                if selected_new is not None:
                    next_states[eid]=state[eid].index_copy(0,token,selected_new)
            state=next_states
            frac=torch.tensor(per_step,device=x.device,dtype=probs.dtype)/max(1,sum(per_step))
            aux.append(len(self.experts)*(probs.mean(0)*frac).sum())
            x=self.norm(x+delta)
            routing.append(logits)
        return x,state,torch.stack(aux).mean(),count,torch.stack(routing,dim=1)


def build_runtime(organization:str,kind:str,dim:int,hidden:int,depth:int,slots:int,experts:int,top_k:int):
    if organization=='sequential':
        return SequentialRuntime(kind,dim,hidden,depth,slots)
    if organization=='corecpu':
        return CoreCPURuntime(kind,dim,hidden,depth,slots,experts,top_k)
    raise ValueError(organization)
