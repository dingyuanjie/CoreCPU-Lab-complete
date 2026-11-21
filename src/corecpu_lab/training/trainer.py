"""One training pipeline for A/B/C/D; configurable learning and evaluation.

Never overwrite an existing run directory unless --resume explicitly points
at a compatible checkpoint. No dataset is silently downloaded.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
import torch
from corecpu_lab.core.config import ModelConfig,TrainConfig
from corecpu_lab.core.utils import seed_all,select_device,write_json,append_jsonl,environment_info,digest_file
from corecpu_lab.models.factory import build_model,architecture_name
from corecpu_lab.training.checkpoint import save_checkpoint,load_checkpoint,restore_training
from corecpu_lab.learning.backprop import GradientLearner
from corecpu_lab.learning.eligibility import EligibilityLearner
from corecpu_lab.tasks.structured.dataset import generator,make_batch
from corecpu_lab.tasks.language.dataset import load_corpus,make_batch as language_batch
from corecpu_lab.benchmarks.evaluate import evaluate_structured,evaluate_language
from corecpu_lab.benchmarks.profile import model_profile


def run_training(model_cfg:ModelConfig,train_cfg:TrainConfig,out_dir:str|Path,resume:str|None=None)->dict:
    model_cfg.validate();train_cfg.validate(model_cfg)
    device=select_device(train_cfg.device)
    seed_all(train_cfg.seed,train_cfg.threads)
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and resume is None:
        raise FileExistsError(f'run directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True,exist_ok=True)
    data_gen=generator(train_cfg.seed+10000)
    model=build_model(model_cfg).to(device)
    learner=(EligibilityLearner(model,model_cfg,train_cfg) if train_cfg.learning=='elig'
             else GradientLearner(model,model_cfg,train_cfg))
    corpus_sha=None
    if model_cfg.task=='language':
        train_data,val_data,corpus_sha=load_corpus(train_cfg.text_path,train_cfg.demo)
        if len(train_data)<=model_cfg.context+1 or len(val_data)<=model_cfg.context+1:
            raise ValueError('corpus split too short for model context')
    start=0
    if resume:
        ckpt=load_checkpoint(resume,device)
        start=restore_training(ckpt,model,learner,model_cfg,train_cfg,data_gen,corpus_sha)
    if train_cfg.steps<start:raise ValueError('steps cannot be smaller than resumed step')
    write_json(out_dir/'config.json',{'model':model_cfg.as_dict(),'train':train_cfg.as_dict(),
                                      'resume':str(resume) if resume else None,'data_sha256':corpus_sha})
    write_json(out_dir/'environment.json',environment_info(device))
    write_json(out_dir/'model_profile.json',model_profile(model))
    log_path=out_dir/'train.jsonl'
    started=time.perf_counter()
    step=start
    for step in range(start+1,train_cfg.steps+1):
        if model_cfg.task=='structured':
            length=int(torch.randint(1,4,(1,),generator=data_gen).item())
            batch=make_batch(train_cfg.batch_size,length,data_gen,'train',device)
        else:
            batch=language_batch(train_data,train_cfg.batch_size,model_cfg.context,data_gen,device)
        metrics=learner.train_step(batch)
        if step==start+1 or step%train_cfg.log_every==0 or step==train_cfg.steps:
            item={'event':'train','step':step,'architecture':architecture_name(model_cfg),**metrics}
            append_jsonl(log_path,item)
            print(json.dumps(item,ensure_ascii=False),flush=True)
        if train_cfg.save_every>0 and step%train_cfg.save_every==0:
            save_checkpoint(out_dir/f'step_{step:07d}.pt',step,model,learner,model_cfg,train_cfg,data_gen,corpus_sha)
    save_checkpoint(out_dir/'checkpoint.pt',step,model,learner,model_cfg,train_cfg,data_gen,corpus_sha)
    if model_cfg.task=='structured':
        eval_results=evaluate_structured(model,device,train_cfg.eval_batch_size,
                                         train_cfg.eval_batches,train_cfg.eval_seed)
    else:
        eval_results=evaluate_language(model,val_data,device,min(8,train_cfg.eval_batch_size),
                                       train_cfg.eval_batches,train_cfg.eval_seed)
    result={
        'architecture':architecture_name(model_cfg),'task':model_cfg.task,
        'learning':train_cfg.learning,'seed':train_cfg.seed,'steps':step,
        'parameters':model_profile(model),'evaluation':eval_results,
        'train_seconds':time.perf_counter()-started,
        'checkpoint':str(out_dir/'checkpoint.pt'),
    }
    write_json(out_dir/'evaluation.json',result)
    append_jsonl(log_path,{'event':'evaluation',**result})
    del learner,model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result
