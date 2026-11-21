"""Factorial model factory; inputs and losses are separate from architecture."""
from __future__ import annotations
from corecpu_lab.core.config import ModelConfig
from corecpu_lab.tasks.structured.model import StructuredModel
from corecpu_lab.tasks.language.model import ByteLanguageModel

ARCHITECTURES={
    'A':('sequential','conventional'),
    'B':('corecpu','conventional'),
    'C':('sequential','sec'),
    'D':('corecpu','sec'),
}

def build_model(config:ModelConfig):
    config.validate()
    return StructuredModel(config) if config.task=='structured' else ByteLanguageModel(config)


def architecture_name(config:ModelConfig)->str:
    for name,(org,cell) in ARCHITECTURES.items():
        if (org,cell)==(config.organization,config.compute):
            return name
    raise ValueError('not part of 2x2 experiment')
