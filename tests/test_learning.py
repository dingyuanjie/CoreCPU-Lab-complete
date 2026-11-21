from dataclasses import replace
import torch
from corecpu_lab.core.config import TrainConfig
from corecpu_lab.models.factory import build_model
from corecpu_lab.learning.backprop import GradientLearner
from corecpu_lab.learning.eligibility import EligibilityLearner
from corecpu_lab.learning.forward_forward import ff_demo
from corecpu_lab.tasks.structured.dataset import make_batch,generator


def test_local_updates_without_temporal_backprop(small_config):
    cfg=small_config('corecpu','sec')
    model=build_model(cfg)
    learner=GradientLearner(model,cfg,TrainConfig(learning='local',steps=1))
    batch=make_batch(4,3,generator(25),'train')
    report=learner.train_step(batch)
    assert torch.isfinite(torch.tensor(report['loss']))
    assert report['grad_norm']>=0


def test_manual_eligibility_changes_head_without_autograd(small_config):
    cfg=small_config('corecpu','sec')
    model=build_model(cfg)
    learner=EligibilityLearner(model,cfg,TrainConfig(learning='elig',steps=1))
    batch=make_batch(4,3,generator(33),'train')
    old=model.head.weight.detach().clone()
    report=learner.train_step(batch)
    assert report['grad_norm']==0
    assert not torch.equal(old,model.head.weight)
    assert all(p.grad is None for p in model.parameters())


def test_ff_standalone_demo():
    output=ff_demo(steps=2)
    assert output['steps']==2
    assert output['loss_start']>0
