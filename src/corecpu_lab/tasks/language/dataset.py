"""Byte-level autoregressive data. Demo is a smoke test, NOT a language benchmark."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch

VOCAB=257
BOS=256
DEMO_TEXT=(
    'CoreCPU builds reusable components. A computation can update state. '
    'A component may be called again. A good baseline is essential. '
    'Research requires honest evaluation and reproducible experiments. '
    'Input changes; knowledge and memory can be external. '\
    'The output is learned, not hard-coded. '\
    'Design systems, test hypotheses, compare fairly.\n'
)*500


def load_corpus(path: str|None,demo: bool):
    if demo:
        if path:raise ValueError('use either demo or text_path')
        data=DEMO_TEXT.encode('utf-8')
    else:
        if not path:raise ValueError('language data requires --text PATH or --demo')
        data=Path(path).read_bytes()
        data.decode('utf-8')  # Reject invalid UTF-8
    if len(data)<256:raise ValueError('corpus too short')
    cut=int(len(data)*0.9)
    return torch.tensor(list(data[:cut]),dtype=torch.long),\
           torch.tensor(list(data[cut:]),dtype=torch.long),\
           hashlib.sha256(data).hexdigest()


def make_batch(data:torch.Tensor,batch:int,context:int,g:torch.Generator,device):
    if len(data)<context+2:raise ValueError('corpus split shorter than context+2')
    start=torch.randint(0,len(data)-context-1,(batch,),generator=g)
    offsets=torch.arange(context)
    x=data[start[:,None]+offsets]
    y=data[start[:,None]+offsets+1]
    return {'tokens':x.to(device),'labels':y.to(device)}
