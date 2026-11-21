from dataclasses import replace
from pathlib import Path
import torch
from corecpu_lab.core.config import ModelConfig,TrainConfig
from corecpu_lab.training.trainer import run_training
from corecpu_lab.training.checkpoint import load_checkpoint,load_inference_model
from corecpu_lab.cli import predict_structured
import json


def test_save_load_predict_and_resume_deterministically(tmp_path):
    cfg=ModelConfig(task='structured',organization='sequential',compute='conventional',
                    hidden_dim=16,ff_dim=24,depth=1,num_experts=2,top_k=1,num_slots=2,heads=4)
    partial=TrainConfig(steps=2,batch_size=4,eval_batches=1,eval_batch_size=4,
                        log_every=1,device='cpu',threads=2)
    folder=tmp_path/'partial'
    run_training(cfg,partial,folder)
    ckpt=load_checkpoint(folder/'checkpoint.pt','cpu')
    assert ckpt['step']==2
    # The exact training-stream model should match uninterrupted training.
    resumed=replace(partial,steps=4)
    run_training(cfg,resumed,folder,resume=folder/'checkpoint.pt')
    direct_dir=tmp_path/'direct'
    run_training(cfg,resumed,direct_dir)
    a=load_checkpoint(folder/'checkpoint.pt','cpu')['model_state']
    b=load_checkpoint(direct_dir/'checkpoint.pt','cpu')['model_state']
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    m,_=load_inference_model(folder/'checkpoint.pt',torch.device('cpu'))
    example=json.loads((Path(__file__).parent.parent/'example.json').read_text())
    result=predict_structured(m,example,torch.device('cpu'))
    assert len(result['step_predictions'])==3
