import pytest
from corecpu_lab.cli import read_config,selftest
from corecpu_lab.core.config import ModelConfig


def test_configs_valid():
    from pathlib import Path
    root=Path(__file__).parent.parent
    for f in (root/'configs').rglob('*.yaml'):
        model,train=read_config(f)
        model.validate();train.validate(model)


def test_no_fake_elig_language():
    with pytest.raises(ValueError):
        from corecpu_lab.core.config import TrainConfig
        TrainConfig(learning='elig').validate(ModelConfig(task='language',compute='sec'))
