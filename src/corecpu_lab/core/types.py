"""Small shared contracts; diagnostics must never leak targets into forward()."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import Tensor

@dataclass
class ModelOutput:
    logits: Tensor                   # [B,T,V] for both tasks
    auxiliary_loss: Tensor
    features: Tensor | None = None  # [B,T,D], for experimental local learning
    attention_logits: Tensor | None = None
    route_logits: Tensor | None = None
    expert_counts: list[int] = field(default_factory=list)
    state: Any = None
    traces: list[Any] = field(default_factory=list)

    def finite(self) -> bool:
        return bool(torch.isfinite(self.logits).all() and torch.isfinite(self.auxiliary_loss).all())

def detach_state(obj: Any) -> Any:
    """Explicitly sever temporal gradients for LOCAL, not for ordinary BP."""
    if isinstance(obj, Tensor):
        return obj.detach()
    if isinstance(obj, list):
        return [detach_state(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(detach_state(x) for x in obj)
    if isinstance(obj, dict):
        return {k: detach_state(v) for k,v in obj.items()}
    return obj


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
