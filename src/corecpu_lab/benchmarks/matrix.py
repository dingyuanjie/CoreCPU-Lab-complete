"""Sequential factorial benchmark, with runs saved even if another run fails."""
from __future__ import annotations
from dataclasses import replace
import csv
import gc
import json
from pathlib import Path
import torch
from corecpu_lab.core.config import ModelConfig,TrainConfig
from corecpu_lab.models.factory import ARCHITECTURES
from corecpu_lab.training.trainer import run_training
from corecpu_lab.core.utils import write_json


def run_matrix(model_base:ModelConfig,train_base:TrainConfig,out_dir:str|Path,
               seeds:list[int],cases:list[str]|None=None)->dict:
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    cases=cases or list(ARCHITECTURES)
    for x in cases:
        if x not in ARCHITECTURES:raise ValueError(f'unknown architecture: {x}')
    csv_file=out_dir/'results.csv'
    if csv_file.exists():raise FileExistsError(f'refusing to overwrite {csv_file}')
    records=[]
    try:
        with csv_file.open('x',newline='',encoding='utf-8') as f:
            fieldnames=['architecture','seed','task','learning','split','accuracy','step_accuracy',
                        'retrieval_accuracy','bits_per_byte','parameters','train_seconds',
                        'examples_per_sec','checkpoint']
            writer=csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            for case in cases:
                org,compute=ARCHITECTURES[case]
                config=replace(model_base,organization=org,compute=compute)
                for seed in seeds:
                    train=replace(train_base,seed=seed)
                    folder=out_dir/f'{case}_{config.task}_{train.learning}_seed{seed}'
                    if folder.exists():raise FileExistsError(f'run already exists: {folder}')
                    print(f'RUN {case} seed={seed}',flush=True)
                    report=run_training(config,train,folder)
                    records.append(report)
                    tests=report['evaluation']
                    splits=(tests.items() if config.task=='structured' else [('validation',tests)])
                    for split,metric in splits:
                        writer.writerow({
                            'architecture':case,'seed':seed,'task':config.task,
                            'learning':train.learning,'split':split,
                            'accuracy':metric.get('final_acc',metric.get('token_accuracy')),
                            'step_accuracy':metric.get('step_acc'),
                            'retrieval_accuracy':metric.get('retrieval_acc'),
                            'bits_per_byte':metric.get('bits_per_byte'),
                            'parameters':report['parameters']['trainable_parameters'],
                            'train_seconds':report['train_seconds'],
                            'examples_per_sec':metric.get('examples_per_sec',metric.get('tokens_per_sec')),
                            'checkpoint':report['checkpoint'],
                        })
                        f.flush()
                    gc.collect()
                    if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:
        write_json(out_dir/'manifest.json',{'completed':records,'planned_cases':cases,'planned_seeds':seeds})
    return {'runs':len(records),'results_csv':str(csv_file),'manifest':str(out_dir/'manifest.json')}
