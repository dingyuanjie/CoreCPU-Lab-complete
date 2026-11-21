
"""Run controlled multi-seed baselines sequentially; preserve every JSONL log."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = {
    'shared': ['--arch','shared'],
    'dense_active': ['--arch','dense','--hidden','96'],
    'dense_total': ['--arch','dense','--hidden','388'],
    'gru': ['--arch','gru'],
    'shared_no_router_label': ['--arch','shared','--route-weight','0'],
    'fixed_route': ['--arch','shared','--fixed-route','--route-weight','0'],
    'top2': ['--arch','shared','--topk','2'],
    'final_only': ['--arch','shared','--step-weight','0'],
    'no_read_loss': ['--arch','shared','--read-weight','0'],
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--steps',type=int,default=700)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seeds',nargs='+',type=int,default=[313,101,202,303])
    p.add_argument('--cases',nargs='+',choices=tuple(CASES),default=list(CASES)[:5])
    p.add_argument('--eval-seed',type=int,default=271828)
    p.add_argument('--eval-batches',type=int,default=50)
    p.add_argument('--out',default='runs/benchmark')
    p.add_argument('--amp',action='store_true')
    a=p.parse_args(); out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    csvpath=out/'results.csv'
    if csvpath.exists(): raise FileExistsError(f'Refusing to overwrite {csvpath}; choose another --out')
    columns=['case','seed','parameters','split','examples','length','final_acc','step_acc','all_steps_acc','retrieval_acc','examples_per_second','assignments_per_step','expert_token_counts','router_top1_matches_op','checkpoint']
    with csvpath.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();f.flush()
        for case in a.cases:
            for seed in a.seeds:
                stem=f'{case}-{seed}'; ck=out/f'{stem}.pt'; log=out/f'{stem}.jsonl'
                cmd=[sys.executable,str(ROOT/'core_cpu.py'),'train-exec',*CASES[case],
                    '--device',a.device,'--steps',str(a.steps),'--batch',str(a.batch),'--threads',str(a.threads),
                    '--seed',str(seed),'--eval-seed',str(a.eval_seed),'--eval-batches',str(a.eval_batches),'--out',str(ck)]
                if a.amp: cmd.append('--amp')
                print('RUN',stem,flush=True)
                if ck.exists() or log.exists(): raise FileExistsError(stem)
                (out/f'{stem}.command.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2),encoding='utf-8')
                with log.open('x',encoding='utf-8') as fp:
                    proc=subprocess.run(cmd,stdout=fp,stderr=subprocess.STDOUT,check=False)
                if proc.returncode: raise RuntimeError(f'{stem} failed: {log}')
                events=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
                report=next(e for e in reversed(events) if e.get('event')=='evaluation')
                for split,metrics in report['metrics'].items():
                    writer.writerow(dict(case=case,seed=seed,parameters=report['parameters'],split=split,checkpoint=str(ck),**metrics))
                f.flush()
    print(f'Finished: {csvpath}',flush=True)

if __name__=='__main__': main()

