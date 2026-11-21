"""CoreCPU-Lab command line: train/eval/predict/generate/benchmark/selftest."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
import yaml
from corecpu_lab.core.config import ModelConfig,TrainConfig
from corecpu_lab.core.utils import seed_all,select_device
from corecpu_lab.training.trainer import run_training
from corecpu_lab.training.checkpoint import load_inference_model
from corecpu_lab.tasks.structured.dataset import OPS,Q
from corecpu_lab.tasks.language.dataset import load_corpus
from corecpu_lab.benchmarks.evaluate import evaluate_structured,evaluate_language
from corecpu_lab.benchmarks.matrix import run_matrix
from corecpu_lab.benchmarks.analysis import analyze_matrix
from corecpu_lab.benchmarks.latency import run_latency_sweep
from corecpu_lab.learning.forward_forward import ff_demo


def read_config(path):
    info=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(info,dict) or 'model' not in info or 'train' not in info:
        raise ValueError('YAML must contain model: and train: mappings')
    return ModelConfig(**info['model']),TrainConfig(**info['train'])


def replace_train(t,cmd):
    vals={}
    for arg,attr in [('steps','steps'),('batch','batch_size'),('seed','seed'),
                      ('learning','learning'),('device','device'),('threads','threads'),
                      ('text','text_path'),('lr','lr'),('eval_batches','eval_batches')]:
        val=getattr(cmd,arg,None)
        if val is not None:vals[attr]=val
    if getattr(cmd,'demo',False):vals.update(demo=True,text_path=None)
    if getattr(cmd,'text',None):vals['demo']=False
    return replace(t,**vals)


def json_print(data):
    print(json.dumps(data,ensure_ascii=False,indent=2,default=str))


def predict_structured(model,spec,device):
    if model.config.task!='structured':raise ValueError('predict needs a structured checkpoint')
    mod=int(spec['modulus'])
    if not 2<=mod<=8:raise ValueError('modulus must be between 2 and 8')
    memory=spec['memory']
    if not isinstance(memory,dict) or not 1<=len(memory)<=16:
        raise ValueError('memory must be a dictionary with 1..16 items')
    keys=list(map(int,memory.keys()))
    vals=list(map(int,memory.values()))
    if any(k<0 or k>=16 for k in keys) or any(v<0 or v>=mod for v in vals):
        raise ValueError('invalid key or value')
    start=int(spec['start_key'])
    if start not in keys:raise ValueError('start_key absent from memory')
    steps=spec['steps']
    if not 1<=len(steps)<=128:raise ValueError('steps must have 1..128 items')
    queries=[start];use_ref=[];literals=[];ops=[]
    for step in steps:
        name=step['op'].upper()
        if name not in OPS:raise ValueError(f'unknown operation: {name}')
        ops.append(OPS.index(name))
        if 'key' in step:
            k=int(step['key'])
            if k not in keys:raise ValueError(f'missing memory key {k}')
            queries.append(k);use_ref.append(True);literals.append(0)
        elif 'literal' in step:
            value=int(step['literal'])
            if not 0<=value<mod:raise ValueError('literal outside modulus')
            queries.append(start);use_ref.append(False);literals.append(value)
        else:raise ValueError('step needs key or literal')
    data={
        'keys':torch.tensor([keys],device=device),
        'values':torch.tensor([vals],device=device),
        'queries':torch.tensor([queries],device=device),
        'use_ref':torch.tensor([use_ref],device=device,dtype=torch.bool),
        'literals':torch.tensor([literals],device=device),
        'ops':torch.tensor([ops],device=device),
        'mod':torch.tensor([mod],device=device),
    }
    model.eval()
    with torch.no_grad():out=model(**data)
    return {'final':int(out.logits[0,-1].argmax(-1).item()),
            'step_predictions':out.logits[0].argmax(-1).tolist(),
            'step_confidence':out.logits[0].softmax(-1).amax(-1).tolist(),
            'retrieved_memory_indices':out.attention_logits[0].argmax(-1).tolist(),
            'expert_token_counts':out.expert_counts,
            'route_argmax':out.route_logits[0].argmax(-1).tolist() if out.route_logits is not None else None,
            'warning':'confidence is not necessarily calibrated; some valid inputs are outside training distribution'}


def selftest():
    from corecpu_lab.models.factory import build_model,ARCHITECTURES
    from corecpu_lab.tasks.structured.dataset import generator,make_batch,only_inputs
    from corecpu_lab.training.losses import structured_loss,language_loss
    seed_all(313,2)
    passed=[]
    for name,(org,cell) in ARCHITECTURES.items():
        cfg=ModelConfig(task='structured',organization=org,compute=cell,hidden_dim=16,
                        ff_dim=24,depth=2,num_experts=3,top_k=1,num_slots=2,heads=4)
        model=build_model(cfg)
        batch=make_batch(3,3,generator(1024),'iid')
        out=model(**only_inputs(batch))
        loss,_=structured_loss(out,batch,cfg)
        assert out.logits.shape==(3,3,8) and torch.isfinite(loss)
        loss.backward()
        assert model.head.weight.grad is not None
        passed.append(name+'_structured')
        cfg2=replace(cfg,task='language',context=8)
        lm=build_model(cfg2)
        xs=torch.randint(0,256,(2,8))
        o=lm(xs)
        assert o.logits.shape==(2,8,257)
        language_loss(o,{'labels':xs},cfg2)[0].backward()
        passed.append(name+'_language')
    json_print({'selftest':'passed','cases':passed})


def parser():
    p=argparse.ArgumentParser(prog='corecpu-lab',description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train',help='train one architecture')
    train.add_argument('--config',required=True)
    train.add_argument('--out',required=True)
    train.add_argument('--resume')
    for sp in (train,):
        sp.add_argument('--steps',type=int)
        sp.add_argument('--batch',type=int)
        sp.add_argument('--seed',type=int)
        sp.add_argument('--lr',type=float)
        sp.add_argument('--learning',choices=('bp','local','elig'))
        sp.add_argument('--device',choices=('auto','cpu','cuda'))
        sp.add_argument('--threads',type=int)
        sp.add_argument('--text')
        sp.add_argument('--demo',action='store_true')
        sp.add_argument('--eval-batches',type=int)
    ev=sub.add_parser('eval',help='evaluate saved checkpoint')
    ev.add_argument('--checkpoint',required=True)
    ev.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ev.add_argument('--batches',type=int,default=20)
    ev.add_argument('--batch',type=int,default=64)
    ev.add_argument('--seed',type=int,default=271828)
    ev.add_argument('--text')
    pr=sub.add_parser('predict',help='structured JSON inference; no target leakage')
    pr.add_argument('--checkpoint',required=True)
    pr.add_argument('--input',required=True)
    pr.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ge=sub.add_parser('generate',help='byte language generation')
    ge.add_argument('--checkpoint',required=True)
    ge.add_argument('--prompt',default='CoreCPU')
    ge.add_argument('--tokens',type=int,default=64)
    ge.add_argument('--temperature',type=float,default=0.8)
    ge.add_argument('--top-k',type=int,default=40)
    ge.add_argument('--seed',type=int,default=313)
    ge.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ma=sub.add_parser('benchmark',help='sequential A/B/C/D factorial benchmark')
    ma.add_argument('--config',required=True)
    ma.add_argument('--out',required=True)
    ma.add_argument('--steps',type=int)
    ma.add_argument('--batch',type=int)
    ma.add_argument('--device',choices=('auto','cpu','cuda'))
    ma.add_argument('--seeds',nargs='+',type=int,default=[313,101,202,303])
    ma.add_argument('--cases',nargs='+',choices=list('ABCD'),default=list('ABCD'))
    ma.add_argument('--learning',choices=('bp','local','elig'))
    ma.add_argument('--eval-batches',type=int)
    ma.add_argument('--text')
    ma.add_argument('--demo',action='store_true')
    an=sub.add_parser('analyze',help='descriptive 2x2 seed-paired interaction report')
    an.add_argument('--csv',required=True)
    an.add_argument('--out')
    pf=sub.add_parser('profile',help='forward-only length vs latency and memory sweep')
    pf.add_argument('--checkpoint',required=True)
    pf.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    pf.add_argument('--lengths',nargs='+',type=int,default=[3,6,12])
    pf.add_argument('--batch',type=int,default=4)
    pf.add_argument('--repeats',type=int,default=10)
    pf.add_argument('--warmup',type=int,default=3)
    sub.add_parser('selftest')
    ff=sub.add_parser('ff-demo')
    ff.add_argument('--steps',type=int,default=30)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='selftest':return selftest()
    if args.command=='ff-demo':return json_print(ff_demo(args.steps))
    if args.command=='analyze':return json_print(analyze_matrix(args.csv,args.out))
    if args.command=='train':
        m,t=read_config(args.config)
        result=run_training(m,replace_train(t,args),args.out,args.resume)
        return json_print(result)
    if args.command=='benchmark':
        m,t=read_config(args.config)
        t=replace_train(t,args)
        result=run_matrix(m,t,args.out,args.seeds,args.cases)
        return json_print(result)
    device=select_device(args.device)
    model,ckpt=load_inference_model(args.checkpoint,device)
    seed_all(getattr(args,'seed',313),2)
    if args.command=='profile':
        return json_print(run_latency_sweep(model,device,args.lengths,args.batch,args.repeats,args.warmup))
    if args.command=='predict':
        spec=json.loads(Path(args.input).read_text(encoding='utf-8'))
        return json_print(predict_structured(model,spec,device))
    if args.command=='generate':
        if model.config.task!='language':raise ValueError('generate needs language checkpoint')
        return print(model.generate(args.prompt,args.tokens,args.temperature,args.top_k))
    if args.command=='eval':
        if model.config.task=='structured':
            return json_print(evaluate_structured(model,device,args.batch,args.batches,args.seed))
        tc=ckpt['train_config']
        path=args.text or tc['text_path']
        train,val,sha=load_corpus(path,bool(tc['demo']) and args.text is None)
        if sha!=ckpt['corpus_sha256']:raise ValueError('eval corpus SHA256 differs from training')
        return json_print(evaluate_language(model,val,device,min(args.batch,8),args.batches,args.seed))

if __name__=='__main__':
    main()
