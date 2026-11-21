"""Strict model and training configuration; JSON-compatible, YAML-serializable."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Literal

Organization = Literal['sequential', 'corecpu']
Compute = Literal['conventional', 'sec']
Task = Literal['structured', 'language']

@dataclass(frozen=True)
class ModelConfig:
    task: Task = 'structured'
    organization: Organization = 'corecpu'
    compute: Compute = 'conventional'
    hidden_dim: int = 64
    ff_dim: int = 96
    depth: int = 3
    num_experts: int = 4
    top_k: int = 1
    num_slots: int = 4
    heads: int = 4
    context: int = 64
    route_weight: float = 0.0
    balance_weight: float = 0.01
    read_weight: float = 0.1
    step_weight: float = 0.5
    max_modulus: int = 8

    def validate(self) -> None:
        if self.task not in ('structured','language'):
            raise ValueError(f'Unknown task: {self.task}')
        if self.organization not in ('sequential','corecpu'):
            raise ValueError(f'Unknown organization: {self.organization}')
        if self.compute not in ('conventional','sec'):
            raise ValueError(f'Unknown compute: {self.compute}')
        for name in ('hidden_dim','ff_dim','depth','num_experts','top_k','num_slots','heads'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be >= 1')
        if self.top_k > self.num_experts:
            raise ValueError('top_k cannot exceed num_experts')
        if self.hidden_dim % self.heads != 0:
            raise ValueError('hidden_dim must be divisible by heads')
        if self.context < 2 or self.max_modulus != 8:
            raise ValueError('context must be >= 2 and v1 structured modulus space must remain 8')
        for name in ('route_weight','balance_weight','read_weight','step_weight'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class TrainConfig:
    learning: str = 'bp'
    seed: int = 313
    steps: int = 100
    batch_size: int = 32
    lr: float = 0.001
    grad_clip: float = 1.0
    eval_batches: int = 4
    eval_batch_size: int = 64
    log_every: int = 20
    save_every: int = 0
    device: str = 'auto'
    threads: int = 2
    amp: bool = False
    text_path: str | None = None
    demo: bool = True
    eval_seed: int = 271828

    def validate(self, model: ModelConfig) -> None:
        if self.learning not in ('bp', 'local', 'elig'):
            raise ValueError(f'learning={self.learning} is not implemented for these models')
        if self.steps < 1 or self.batch_size < 1 or self.lr <= 0 or self.threads < 1:
            raise ValueError('steps/batch/threads must be positive and lr > 0')
        if self.eval_batches < 1 or self.eval_batch_size < 1 or self.log_every < 1:
            raise ValueError('eval_batches/eval_batch_size/log_every must be >= 1')
        if self.save_every < 0 or self.grad_clip <= 0:
            raise ValueError('save_every must be >= 0, grad_clip > 0')
        if self.learning == 'local' and model.task != 'structured':
            raise ValueError('local learner currently supports the structured task only')
        if self.learning == 'elig' and (model.compute != 'sec' or model.task != 'structured'):
            raise ValueError('elig learner is an experimental SEC+structured-only learning path')
        if self.amp and self.learning != 'bp':
            raise ValueError('AMP is initially supported only for BP learner')

    def as_dict(self) -> dict:
        return asdict(self)
