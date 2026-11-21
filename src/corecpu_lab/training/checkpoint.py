"""Atomic checkpoints with full RNG and optimizer/manual learner state.

Load only checkpoint files you trust: PyTorch deserialization is not a sandbox.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
import torch
from corecpu_lab.core.config import ModelConfig

CHECKPOINT_VERSION=1

def save_checkpoint(path,step,model,learner,model_cfg,train_cfg,generator,corpus_sha256=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        'format':'CoreCPU-Lab','version':CHECKPOINT_VERSION,'step':step,
        'model_config':model_cfg.as_dict(),
        'train_config':train_cfg.as_dict(),
        'corpus_sha256':corpus_sha256,
        'model_state':model.state_dict(),
        'learner_state':learner.state_dict(),
        'rng':{'torch':torch.get_rng_state(),
               'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               'python':random.getstate(),
               'data_generator':generator.get_state()},
    }
    tmp=path.with_name(path.name+'.tmp')
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def load_checkpoint(path,device='cpu')->dict:
    data=torch.load(path,map_location=device,weights_only=False)
    if data.get('format')!='CoreCPU-Lab' or data.get('version')!=CHECKPOINT_VERSION:
        raise ValueError('not a compatible CoreCPU-Lab checkpoint')
    return data


def restore_training(data,model,learner,model_cfg,train_cfg,generator,corpus_sha256):
    if data['model_config']!=model_cfg.as_dict():
        raise ValueError('model configuration mismatch when resuming')
    original=data['train_config']
    for key in ('learning','batch_size','lr','seed','demo','text_path','grad_clip'):
        if original[key]!=getattr(train_cfg,key):
            raise ValueError(f'resume requires original train.{key}')
    if data['corpus_sha256']!=corpus_sha256:
        raise ValueError('corpus hash mismatch; refusing to resume')
    model.load_state_dict(data['model_state'],strict=True)
    learner.load_state_dict(data['learner_state'])
    state=data['rng']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    random.setstate(state['python'])
    generator.set_state(state['data_generator'].cpu())
    return int(data['step'])


def load_inference_model(path,device):
    from corecpu_lab.models.factory import build_model
    ckpt=load_checkpoint(path,device)
    cfg=ModelConfig(**ckpt['model_config'])
    model=build_model(cfg).to(device)
    model.load_state_dict(ckpt['model_state'],strict=True)
    model.eval()
    return model,ckpt
