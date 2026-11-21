from __future__ import annotations
import pytest
import torch
from corecpu_lab.core.config import ModelConfig

@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(313)

@pytest.fixture
def small_config():
    def make(org='corecpu',compute='conventional',task='structured'):
        return ModelConfig(task=task,organization=org,compute=compute,hidden_dim=16,
                           ff_dim=24,depth=2,num_experts=3,top_k=1,num_slots=2,
                           heads=4,context=8)
    return make
