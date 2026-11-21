"""All losses live outside the model. Ground truth never enters forward()."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from corecpu_lab.core.types import ModelOutput
from corecpu_lab.core.config import ModelConfig


def structured_loss(out:ModelOutput,batch:dict,cfg:ModelConfig, *, local:bool=False):
    logits=out.logits
    labels=batch['labels']
    ce=F.cross_entropy(logits.flatten(0,1),labels.flatten())
    end=F.cross_entropy(logits[:,-1],labels[:,-1])
    # Local mode optimizes step-wise losses with truncated temporal gradients.
    score=ce if local else end+cfg.step_weight*ce
    reader_loss=score.new_zeros(())
    if out.attention_logits is not None:
        mask=torch.cat((torch.ones_like(batch['use_ref'][:,:1]),batch['use_ref']),dim=1)
        reader_loss=F.cross_entropy(out.attention_logits[mask],batch['qpos'][mask])
    route_loss=score.new_zeros(())
    if cfg.route_weight>0 and out.route_logits is not None and cfg.num_experts==4:
        route_loss=F.cross_entropy(out.route_logits.flatten(0,1),batch['ops'].flatten())
    loss=score+cfg.read_weight*reader_loss+cfg.route_weight*route_loss+cfg.balance_weight*out.auxiliary_loss
    return loss,{
        'loss':float(loss.detach()), 'loss_final':float(end.detach()),
        'loss_steps':float(ce.detach()), 'loss_read':float(reader_loss.detach()),
        'loss_route':float(route_loss.detach()), 'balance':float(out.auxiliary_loss.detach()),
        'accuracy':float((logits[:,-1].argmax(-1)==labels[:,-1]).float().mean().detach()),
    }


def language_loss(out:ModelOutput,batch:dict,cfg:ModelConfig):
    ce=F.cross_entropy(out.logits.flatten(0,1),batch['labels'].flatten())
    loss=ce+cfg.balance_weight*out.auxiliary_loss
    return loss,{'loss':float(loss.detach()),'cross_entropy_nats':float(ce.detach()),
                 'bits_per_byte':float((ce.detach()/0.6931471805599453)),
                 'accuracy':float((out.logits.argmax(-1)==batch['labels']).float().mean().detach()),
                 'balance':float(out.auxiliary_loss.detach())}
