"""Seed-paired analysis of a 2x2 experiment CSV.

This is descriptive, not a statistical significance test. Pairing ensures
that A/B/C/D use the same evaluation random seeds and split labels.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
from corecpu_lab.core.utils import write_json


def _stats(xs):
    return {'n':len(xs),'mean':mean(xs) if xs else None,
            'sd':stdev(xs) if len(xs)>=2 else None,'values':xs}


def analyze_matrix(csv_file:str|Path,out_file:str|Path|None=None)->dict:
    with Path(csv_file).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    cells:dict[str,dict[str,dict[int,float]]]=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        if row['architecture'] not in ('A','B','C','D'):raise ValueError('invalid architecture')
        split=row['split'];arch=row['architecture'];seed=int(row['seed'])
        metric=row['accuracy']
        if not metric:continue
        if seed in cells[split][arch]:
            raise ValueError(f'duplicate experiment split={split} arch={arch} seed={seed}')
        cells[split][arch][seed]=float(metric)
    result={}
    for split,by_arch in sorted(cells.items()):
        means={arch:_stats([v for _,v in sorted(by_arch.get(arch,{}).items())])
               for arch in 'ABCD'}
        common=set.intersection(*(set(by_arch.get(arch,{})) for arch in 'ABCD'))
        effects={'organization':[],'compute':[],'interaction':[]}
        for seed in sorted(common):
            a,b,c,d=[by_arch[arch][seed] for arch in 'ABCD']
            effects['organization'].append(((b-a)+(d-c))/2)
            effects['compute'].append(((c-a)+(d-b))/2)
            effects['interaction'].append(d-b-c+a)
        result[split]={
            'architecture':means,
            'paired_seeds':sorted(common),
            'effects':{k:_stats(v) for k,v in effects.items()},
            'note':'Descriptive seed pairing only; not causal proof or a significance test',
        }
    payload={'source':str(csv_file),'splits':result}
    if out_file is not None:write_json(out_file,payload)
    return payload
