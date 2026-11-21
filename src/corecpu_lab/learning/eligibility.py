"""Manual three-factor SEC learning baseline, not a claim of exact e-prop.

No backward(), torch.autograd.grad() or optimizer. Only the classifier head
is directly supervised; SEC output projections receive a centered scalar
feedback times activity eligibility. Other SEC/gate/router/reader parameters
remain fixed. It is a genuine runnable no-global-gradient ablation, not an
end-to-end replacement for BP. This limitation is explicit in documentation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from corecpu_lab.compute.cells import StateEvolutionCell
from corecpu_lab.tasks.structured.dataset import only_inputs

class EligibilityLearner:
    def __init__(self,model,model_cfg,train_cfg):
        self.model,self.cfg,self.tcfg=model,model_cfg,train_cfg
        self.cells=[x for x in model.modules() if isinstance(x,StateEvolutionCell)]
        if not self.cells:raise ValueError('eligibility learning needs SEC cells')
        self.steps=0

    @torch.no_grad()
    def train_step(self,batch:dict)->dict:
        self.model.train()
        for cell in self.cells:cell.begin_local()
        try:
            output=self.model(**only_inputs(batch))
            if not output.finite():raise FloatingPointError('nonfinite SEC logits')
            logits=output.logits.float()
            target=batch['labels']
            probs=logits.softmax(-1)
            err=probs-F.one_hot(target,logits.shape[-1]).float()
            features=output.features.detach()
            grad_w=torch.einsum('btq,btd->qd',err,features)/(target.numel())
            grad_b=err.mean((0,1))
            lr=self.tcfg.lr
            self.model.head.weight.sub_(lr*grad_w.clamp(-self.tcfg.grad_clip,self.tcfg.grad_clip))
            self.model.head.bias.sub_(lr*grad_b.clamp(-self.tcfg.grad_clip,self.tcfg.grad_clip))
            per_example=(logits.argmax(-1)==target).float().mean(1)
            feedback=(per_example-per_example.mean()).clamp(-1,1)
            for cell in self.cells:
                cell.apply_local_feedback(feedback,lr=lr)
            self.steps+=1
            loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
            return {'loss':float(loss),'accuracy':float((logits[:,-1].argmax(-1)==target[:,-1]).float().mean()),
                    'manual_learning':'classifier_gradient_plus_scalar_three_factor_SEC',
                    'grad_norm':0.0}
        finally:
            for cell in self.cells:
                cell._local_records.clear()
                cell._collect_local=False

    def state_dict(self):
        return {'manual_steps':self.steps}
    def load_state_dict(self,state:dict):
        self.steps=int(state.get('manual_steps',0))
