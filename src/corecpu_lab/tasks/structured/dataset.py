"""Matches the historic CoreCPU v0.1 dataset recipe and held-out splits.

Only this module computes labels. Model forward() accepts INPUT_KEYS, never labels.
"""
from __future__ import annotations
import torch

Q, N_OPS, N_KEYS, N_FACTS = 8, 4, 16, 8
OPS = ('ADD', 'SUB', 'MUL', 'MAX')
HELD_PAIRS = ((0,2),(1,3))
INPUT_KEYS = ('keys','values','queries','use_ref','literals','ops','mod')


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device='cpu').manual_seed(seed)


def _rand(high: int, shape: tuple, g: torch.Generator) -> torch.Tensor:
    return torch.randint(high,shape,generator=g)


def _forbidden(prev: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    return ((prev==0)&(nxt==2)) | ((prev==1)&(nxt==3))


def make_batch(batch_size: int, length: int, g: torch.Generator,
               split: str = 'train', device: str | torch.device = 'cpu') -> dict[str,torch.Tensor]:
    if length < 1 or batch_size < 1 or split not in ('train','iid','pair','long','mod'):
        raise ValueError('invalid split/length/batch_size')
    if split=='pair' and length<2:
        raise ValueError('pair split needs length >= 2')
    b=batch_size
    mods=torch.tensor((5,7) if split=='mod' else (4,6,8))
    mod=mods[_rand(len(mods),(b,),g)]
    m=mod[:,None]
    keys=torch.rand((b,N_KEYS),generator=g).argsort(-1)[:,:N_FACTS]
    values=(torch.rand((b,N_FACTS),generator=g)*m).long()
    qpos=_rand(N_FACTS,(b,length+1),g)
    queries=keys.gather(1,qpos)
    qvals=values.gather(1,qpos)
    use_ref=torch.rand((b,length),generator=g)<0.5
    literals=(torch.rand((b,length),generator=g)*m).long()
    args=torch.where(use_ref,qvals[:,1:],literals)
    ops=_rand(N_OPS,(b,length),g)
    for t in range(1,length):
        bad=_forbidden(ops[:,t-1],ops[:,t])
        while bool(bad.any()):
            ops[:,t]=torch.where(bad,_rand(N_OPS,(b,),g),ops[:,t])
            bad=_forbidden(ops[:,t-1],ops[:,t])
    if split=='pair':
        pos=_rand(length-1,(b,),g)
        pair=torch.tensor(HELD_PAIRS)[_rand(len(HELD_PAIRS),(b,),g)]
        idx=torch.arange(b)
        ops[idx,pos],ops[idx,pos+1]=pair[:,0],pair[:,1]
    s=qvals[:,0]
    labels=[]
    for t in range(length):
        a,op=args[:,t],ops[:,t]
        s=torch.where(op==0,(s+a)%mod,
            torch.where(op==1,(s-a)%mod,
              torch.where(op==2,(s*a)%mod,torch.maximum(s,a))))
        labels.append(s)
    batch=dict(keys=keys, values=values, queries=queries, use_ref=use_ref,
               literals=literals,ops=ops,mod=mod,labels=torch.stack(labels,1),qpos=qpos)
    return {k:v.to(device) for k,v in batch.items()}


def only_inputs(batch: dict) -> dict:
    """Separate targets and secret address labels from the model by construction."""
    return {k:batch[k] for k in INPUT_KEYS}
