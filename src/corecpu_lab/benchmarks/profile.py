"""Model cost report. FLOPs estimates are not claimed as hardware counter values."""
from __future__ import annotations
from corecpu_lab.core.types import count_parameters

def model_profile(model):
    return {
        'trainable_parameters':count_parameters(model),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'param_fp32_bytes':sum(p.numel() for p in model.parameters())*4,
        'estimated_adamw_param_optimizer_grad_bytes':sum(p.numel() for p in model.parameters())*16,
        'warning':'estimates exclude activations, CUDA workspace and allocator overhead',
    }
