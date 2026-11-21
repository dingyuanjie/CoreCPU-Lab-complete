"""Warm-up-based forward-only latency sweep with explicit CUDA synchronization."""
from __future__ import annotations
import time
import torch
from corecpu_lab.core.utils import sync
from corecpu_lab.tasks.structured.dataset import generator,make_batch,only_inputs

@torch.no_grad()
def run_latency_sweep(model,device,lengths:list[int],batch_size:int=4,repeats:int=10,warmup:int=3):
    if batch_size<1 or repeats<1 or warmup<0:raise ValueError('bad profile options')
    model.eval()
    result=[]
    for length in lengths:
        if length<1:raise ValueError('length must be >=1')
        if model.config.task=='language':
            if length>model.config.context:raise ValueError('language length exceeds context')
            inputs={'tokens':torch.randint(0,256,(batch_size,length),device=device)}
        else:
            batch=make_batch(batch_size,length,generator(313),'iid',device)
            inputs=only_inputs(batch)
        for _ in range(warmup):model(**inputs)
        sync(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        for _ in range(repeats):model(**inputs)
        sync(device)
        elapsed=time.perf_counter()-started
        result.append({'length':length,'batch_size':batch_size,'repeats':repeats,
                       'ms_per_forward':elapsed/repeats*1000,
                       'input_units_per_sec':batch_size*length*repeats/max(elapsed,1e-9),
                       'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                       'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                       'quality_not_measured':True})
    return result
