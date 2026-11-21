"""Ordinary BP and explicitly truncated local-step supervised learning."""
from __future__ import annotations
import torch
from corecpu_lab.core.config import ModelConfig,TrainConfig
from corecpu_lab.tasks.structured.dataset import only_inputs
from corecpu_lab.training.losses import structured_loss,language_loss

class GradientLearner:
    def __init__(self,model:torch.nn.Module,model_cfg:ModelConfig,train_cfg:TrainConfig):
        self.model,self.model_cfg,self.train_cfg=model,model_cfg,train_cfg
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=train_cfg.lr)
        if train_cfg.amp:
            if next(model.parameters()).device.type!='cuda' or not torch.cuda.is_bf16_supported():
                raise ValueError('AMP requires a CUDA GPU with BF16 support')
        self.use_amp=train_cfg.amp
        self.local=train_cfg.learning=='local'

    def train_step(self,batch:dict)->dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        device=next(self.model.parameters()).device
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=self.use_amp):
            if self.model_cfg.task=='structured':
                out=self.model(**only_inputs(batch),local_detach=self.local)
                loss,metrics=structured_loss(out,batch,self.model_cfg,local=self.local)
            else:
                out=self.model(batch['tokens'])
                loss,metrics=language_loss(out,batch,self.model_cfg)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_cfg.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError('nonfinite gradient norm')
        self.optimizer.step()
        metrics['grad_norm']=float(grad_norm)
        return metrics

    def state_dict(self):
        return {'optimizer':self.optimizer.state_dict()}
    def load_state_dict(self,state:dict):
        self.optimizer.load_state_dict(state['optimizer'])
