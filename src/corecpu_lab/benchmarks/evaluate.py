"""Repeatable held-out evaluation and sequential latency measurement."""
from __future__ import annotations
import math
import time
import torch
import torch.nn.functional as F
from corecpu_lab.tasks.structured.dataset import make_batch,only_inputs,generator
from corecpu_lab.tasks.language.dataset import make_batch as language_batch
from corecpu_lab.core.utils import sync

@torch.no_grad()
def evaluate_structured(model,device:torch.device,batch_size:int=64,batches:int=4,seed:int=271828)->dict:
    old_mode=model.training
    model.eval()
    reports={}
    try:
        for split,length in (('iid',3),('pair',3),('long',6),('mod',3)):
            g=generator(seed)
            all_correct=last_correct=step_correct=0
            reads=read_correct=0
            route_counts=[0]*model.config.num_experts if model.config.organization=='corecpu' else []
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            sync(device);t0=time.perf_counter()
            for _ in range(batches):
                batch=make_batch(batch_size,length,g,split,device)
                out=model(**only_inputs(batch))
                if not out.finite():raise FloatingPointError(f'nonfinite eval logits in {split}')
                match=out.logits.argmax(-1)==batch['labels']
                all_correct+=int(match.all(-1).sum().item())
                last_correct+=int(match[:,-1].sum().item())
                step_correct+=int(match.sum().item())
                mask=torch.cat((torch.ones_like(batch['use_ref'][:,:1]),batch['use_ref']),1)
                read_correct+=int(((out.attention_logits.argmax(-1)==batch['qpos']) & mask).sum().item())
                reads+=int(mask.sum().item())
                if route_counts:route_counts=[a+b for a,b in zip(route_counts,out.expert_counts)]
            sync(device)
            elapsed=max(1e-9,time.perf_counter()-t0)
            n=batch_size*batches
            reports[split]={
                'examples':n,'length':length,
                'final_acc':last_correct/n,
                'step_acc':step_correct/(n*length),
                'all_steps_acc':all_correct/n,
                'retrieval_acc':read_correct/max(1,reads),
                'examples_per_sec':n/elapsed,'elapsed_seconds':elapsed,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                'expert_token_counts':route_counts,
            }
    finally:model.train(old_mode)
    return reports

@torch.no_grad()
def evaluate_language(model,data,device,batch_size=4,batches=4,seed=271828)->dict:
    old_mode=model.training
    model.eval()
    g=generator(seed)
    nats=0.0;correct=0;total=0
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    sync(device);t0=time.perf_counter()
    try:
        for _ in range(batches):
            batch=language_batch(data,batch_size,model.config.context,g,device)
            output=model(batch['tokens'])
            logits=output.logits.flatten(0,1)
            labels=batch['labels'].flatten()
            nats+=float(F.cross_entropy(logits,labels,reduction='sum').item())
            correct+=int((logits.argmax(-1)==labels).sum().item())
            total+=labels.numel()
    finally:model.train(old_mode)
    sync(device)
    ce=nats/max(1,total)
    return {'tokens':total,'cross_entropy_nats':ce,'bits_per_byte':ce/math.log(2),
            'perplexity_per_byte':math.exp(min(ce,50)),
            'token_accuracy':correct/max(1,total),
            'tokens_per_sec':total/max(1e-9,time.perf_counter()-t0),
            'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
