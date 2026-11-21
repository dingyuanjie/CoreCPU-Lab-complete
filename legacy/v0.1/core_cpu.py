
"""CoreCPU v0.1: runnable PyTorch research prototype. No external dependencies except torch."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

Q, N_OPS, N_KEYS, N_SLOTS = 8, 4, 16, 8
OPS = ('ADD', 'SUB', 'MUL', 'MAX')
SEEN_MODS, NEW_MODS = (4, 6, 8), (5, 7)
HELD_PAIRS = ((0, 2), (1, 3))
BYTE_VOCAB, BOS = 257, 256


def seed_everything(seed: int, threads: int = 4):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


# ------------------ Reusable neural components ------------------
class MLP(nn.Module):
    def __init__(self, d, h, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, out))

    def forward(self, x):
        return self.net(x)


class SparseBank(nn.Module):
    """Real token-wise dispatch; only selected experts execute. All weights resident.

    Returns output, router logits, differentiable load penalty, expert-token counts.
    Top-1 uses a straight-through selected probability; top-k uses normalized gates.
    """
    def __init__(self, d, h, out, experts=4, topk=2):
        super().__init__()
        if not 1 <= topk <= experts:
            raise ValueError('Require 1 <= topk <= experts')
        self.experts, self.topk, self.out = experts, topk, out
        self.router = nn.Linear(d, experts)
        self.bank = nn.ModuleList(MLP(d, h, out) for _ in range(experts))

    def forward(self, x, route_input=None, fixed_ids=None):
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        rf = flat if route_input is None else route_input.reshape_as(flat)
        logits = self.router(rf).float()
        prob = logits.softmax(-1)
        if fixed_ids is None:
            chosen_p, chosen = prob.topk(self.topk, dim=-1)
            if self.topk == 1:
                weights = chosen_p / chosen_p.detach().clamp_min(1e-12)
            else:
                weights = chosen_p / chosen_p.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            chosen = fixed_ids.reshape(-1, 1).long()
            weights = torch.ones_like(chosen, dtype=prob.dtype)
        out = flat.new_zeros((flat.shape[0], self.out))
        counts = []
        for eid, expert in enumerate(self.bank):
            token, rank = (chosen == eid).nonzero(as_tuple=True)
            counts.append(token.numel())
            if token.numel():
                value = expert(flat.index_select(0, token))
                weighted = value * weights[token, rank, None].to(value.dtype)
                out = out.index_add(0, token, weighted.to(out.dtype))
        frac = prob.new_tensor(counts) / max(sum(counts), 1)
        balance = self.experts * (prob.mean(0) * frac).sum()
        return out.reshape(*shape, self.out), logits.reshape(*shape, self.experts), balance, counts


class ExternalReader(nn.Module):
    """Addressable external facts. Shared learned key embedding; values are runtime data."""
    def __init__(self, key_dim=24):
        super().__init__()
        self.keys = nn.Embedding(N_KEYS, key_dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, keys, values, queries):
        # keys,values [B,M]; queries [B,R]. All queries refer to a present key.
        k = F.normalize(self.keys(keys).float(), dim=-1)
        q = F.normalize(self.keys(queries).float(), dim=-1)
        logits = (q @ k.transpose(-1, -2)).float() * self.log_scale.exp().clamp(max=80)
        value_dist = F.one_hot(values, Q).float()
        read = (logits.softmax(-1) @ value_dist).float()
        return read, logits


# ------------------ Controlled executor dataset ------------------
def randints(high, shape, g):
    return torch.randint(high, shape, generator=g)


def forbidden(prev, nxt):
    return ((prev == 0) & (nxt == 2)) | ((prev == 1) & (nxt == 3))


def make_batch(b, length, g, split='train', device='cpu'):
    """Online data, no target labels enter model input. Held-out pair and modulus splits.

    State start = external value. Each operation consumes literal or external value.
    All arithmetic is in [0, modulus); only the generator knows exact arithmetic.
    """
    if length < 1 or b < 1 or split not in ('train', 'iid', 'pair', 'long', 'mod'):
        raise ValueError('Invalid split, batch size or length')
    mods = torch.tensor(NEW_MODS if split == 'mod' else SEEN_MODS)
    mod = mods[randints(len(mods), (b,), g)]
    m = mod[:, None]
    keys = torch.rand((b, N_KEYS), generator=g).argsort(-1)[:, :N_SLOTS]
    # randint Q then remainder would bias modulus 5/6/7; sample uniform explicitly.
    vals = (torch.rand((b, N_SLOTS), generator=g) * m).long()
    qpos = randints(N_SLOTS, (b, length + 1), g)
    queries = keys.gather(1, qpos)
    qvals = vals.gather(1, qpos)
    use_ref = torch.rand((b, length), generator=g) < 0.5
    literals = (torch.rand((b, length), generator=g) * m).long()
    operands = torch.where(use_ref, qvals[:, 1:], literals)
    ops = randints(N_OPS, (b, length), g)
    # For every split except pair, exclude both held-out adjacent operator pairs.
    for t in range(1, length):
        bad = forbidden(ops[:, t-1], ops[:, t])
        while bool(bad.any()):
            ops[:, t] = torch.where(bad, randints(N_OPS, (b,), g), ops[:, t])
            bad = forbidden(ops[:, t-1], ops[:, t])
    if split == 'pair':
        if length < 2:
            raise ValueError('pair split needs length >= 2')
        pos = randints(length-1, (b,), g)
        pair = torch.tensor(HELD_PAIRS)[randints(len(HELD_PAIRS), (b,), g)]
        row = torch.arange(b)
        ops[row, pos], ops[row, pos+1] = pair[:, 0], pair[:, 1]
    s = qvals[:, 0]
    labels = []
    for t in range(length):
        a, op = operands[:, t], ops[:, t]
        s = torch.where(op == 0, (s+a) % mod,
            torch.where(op == 1, (s-a) % mod,
            torch.where(op == 2, (s*a) % mod, torch.maximum(s, a))))
        labels.append(s)
    return to_device(dict(keys=keys, values=vals, queries=queries, qpos=qpos,
        use_ref=use_ref, literals=literals, ops=ops, mod=mod,
        labels=torch.stack(labels, 1)), device)


def model_inputs(batch):
    return {k: batch[k] for k in ('keys', 'values', 'queries', 'use_ref', 'literals', 'ops', 'mod')}


class Executor(nn.Module):
    """Structured IR -> external read -> reusable recurrent mechanisms -> categorical state."""
    def __init__(self, arch='shared', d=64, hidden=96, experts=4, topk=1,
                 fixed_route=False):
        super().__init__()
        if arch not in ('shared', 'dense', 'gru'):
            raise ValueError(arch)
        if fixed_route and (arch != 'shared' or experts != N_OPS):
            raise ValueError('fixed_route requires shared architecture and four experts')
        self.arch, self.experts, self.fixed_route = arch, experts, fixed_route
        self.reader = ExternalReader()
        self.op = nn.Embedding(N_OPS, d)
        self.project = nn.Sequential(nn.Linear(2*Q+1, d), nn.GELU(), nn.LayerNorm(d))
        if arch == 'shared':
            self.mechanism = SparseBank(d, hidden, Q, experts, topk)
        elif arch == 'dense':
            self.mechanism = MLP(d, hidden, Q)
        else:
            self.init_h = nn.Linear(Q, d)
            self.gru = nn.GRUCell(d, d)
            self.head = MLP(d, hidden, Q)

    def forward(self, keys, values, queries, use_ref, literals, ops, mod):
        read, attention = self.reader(keys, values, queries)
        state = read[:, 0]
        args = torch.where(use_ref[..., None], read[:, 1:], F.one_hot(literals, Q).float())
        cap = mod.float()[:, None] / Q
        invalid = torch.arange(Q, device=mod.device)[None, :] >= mod[:, None]
        h = self.init_h(state) if self.arch == 'gru' else None
        outputs, routes, balances = [], [], []
        counts = [0] * self.experts if self.arch == 'shared' else []
        for t in range(ops.shape[1]):
            x = self.project(torch.cat((state, args[:, t], cap), -1))
            conditioned = x + self.op(ops[:, t])
            if self.arch == 'shared':
                # Operation identifier influences both router and expert computation.
                fixed = ops[:, t] if self.fixed_route else None
                logits, r, aux, c = self.mechanism(conditioned, fixed_ids=fixed)
                routes.append(r); balances.append(aux)
                counts = [a+b for a, b in zip(counts, c)]
            elif self.arch == 'dense':
                logits = self.mechanism(conditioned)
            else:
                h = self.gru(conditioned, h)
                logits = self.head(h)
            # Known output type/range; no hard-coded arithmetic implementation.
            logits = logits.float().masked_fill(invalid, -1e4)
            state = logits.softmax(-1)  # no teacher forcing; gradients cross all steps
            outputs.append(logits)
        return dict(logits=torch.stack(outputs, 1), attention=attention,
            route=torch.stack(routes, 1) if routes else None,
            balance=torch.stack(balances).mean() if balances else state.new_zeros(()),
            counts=counts)


def exec_loss(model, batch, step_weight=0.5, read_weight=0.1, route_weight=0.1, balance_weight=0.01):
    o = model(**model_inputs(batch))
    logits, labels = o['logits'], batch['labels']
    final = F.cross_entropy(logits[:, -1], labels[:, -1])
    steps = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    mask = torch.cat((torch.ones_like(batch['use_ref'][:, :1]), batch['use_ref']), 1)
    mem = F.cross_entropy(o['attention'][mask], batch['qpos'][mask])
    route = final.new_zeros(())
    if o['route'] is not None and model.experts == N_OPS and not model.fixed_route and route_weight:
        route = F.cross_entropy(o['route'].flatten(0, 1), batch['ops'].flatten())
    # Fixed routing has no learned dispatch to balance.
    bal = o['balance'] if not model.fixed_route else final.new_zeros(())
    loss = final + step_weight*steps + read_weight*mem + route_weight*route + balance_weight*bal
    return loss, o


@torch.no_grad()
def evaluate_executor(model, device, batch_size, batches, seed):
    model.eval()
    report = {}
    for split, length in [('iid', 3), ('pair', 3), ('long', 6), ('mod', 3)]:
        g = rng(seed)  # fixed independent generator; same examples across architectures
        correct = tokens = all_correct = count = reads = good_reads = 0
        routing_correct = route_total = 0
        route_counts = [0] * model.experts if model.arch == 'shared' else []
        sync(device); t0 = time.perf_counter()
        for _ in range(batches):
            batch = make_batch(batch_size, length, g, split, device)
            o = model(**model_inputs(batch))
            match = o['logits'].argmax(-1) == batch['labels']
            correct += int(match[:, -1].sum()); tokens += int(match.sum())
            all_correct += int(match.all(1).sum()); count += batch_size
            mask = torch.cat((torch.ones_like(batch['use_ref'][:, :1]), batch['use_ref']), 1)
            good_reads += int(((o['attention'].argmax(-1) == batch['qpos']) & mask).sum())
            reads += int(mask.sum())
            if route_counts:
                route_counts = [a+b for a,b in zip(route_counts, o['counts'])]
                if model.experts == N_OPS and not model.fixed_route:
                    routing_correct += int((o['route'].argmax(-1) == batch['ops']).sum())
                    route_total += batch_size*length
        sync(device)
        item = dict(examples=count, length=length, final_acc=correct/count,
            step_acc=tokens/(count*length), all_steps_acc=all_correct/count,
            retrieval_acc=good_reads/reads, examples_per_second=count/max(time.perf_counter()-t0,1e-9))
        if route_counts:
            item.update(expert_token_counts=route_counts,
                        assignments_per_step=sum(route_counts)/(count*length))
            if route_total:
                item['router_top1_matches_op'] = routing_correct/route_total
        report[split] = item
    return report


# ------------------ Byte language model ------------------
class CausalAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by heads')
        self.heads, self.head_dim = heads, d//heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b,t,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        z = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(z.transpose(1,2).contiguous().reshape(b,t,d))


class LMBlock(nn.Module):
    def __init__(self,d,heads,hidden,experts,topk,sparse):
        super().__init__()
        self.a_norm, self.f_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttention(d,heads)
        self.sparse = sparse
        self.ff = SparseBank(d,hidden,d,experts,topk) if sparse else MLP(d,hidden,d)

    def forward(self,x):
        x = x + self.attn(self.a_norm(x))
        if self.sparse:
            y, _, aux, c = self.ff(self.f_norm(x))
        else:
            y, aux, c = self.ff(self.f_norm(x)), x.new_zeros(()), []
        return x+y, aux, c


class CoreLM(nn.Module):
    """Causal byte LM; recurrent shared block or untied dense Transformer control.

    Shared = one attention+expert bank reused at every depth. This is separate
    from the executor experiment: they share code/ideas, NOT trained weights.
    """
    def __init__(self, arch='shared', d=128, hidden=256, heads=4, depth=4,
                 experts=4, topk=2, context=128):
        super().__init__()
        if arch not in ('shared','shared_dense','dense') or depth < 1 or context < 2:
            raise ValueError('Invalid LM architecture')
        self.arch, self.depth, self.context, self.experts = arch, depth, context, experts
        self.token = nn.Embedding(BYTE_VOCAB,d)
        self.pos = nn.Embedding(context,d)
        self.depth_emb = nn.Embedding(depth,d)
        self.blocks = nn.ModuleList(LMBlock(d,heads,hidden,experts,topk,arch=='shared')
                                    for _ in range(depth if arch=='dense' else 1))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,BYTE_VOCAB,bias=False)
        self.head.weight = self.token.weight
        # Tied embeddings must not start with standard deviation 1 (oversized logits).
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.depth_emb.weight, std=0.02)

    def forward(self, ids):
        t = ids.shape[1]
        if not 1 <= t <= self.context:
            raise ValueError('Input length exceeds context')
        x = self.token(ids) + self.pos(torch.arange(t,device=ids.device))[None]
        auxs = []; counts = [0]*self.experts if self.arch=='shared' else []
        for i in range(self.depth):
            block = self.blocks[i if self.arch=='dense' else 0]
            x, aux, c = block(x + self.depth_emb.weight[i],)
            auxs.append(aux)
            if counts:
                counts = [a+b for a,b in zip(counts,c)]
        return self.head(self.norm(x)).float(), torch.stack(auxs).mean(), counts

    @torch.no_grad()
    def generate(self, prompt, tokens=100, temperature=0.8, top_k=40, seed=313):
        if tokens < 0 or temperature < 0 or not 1 <= top_k <= BYTE_VOCAB:
            raise ValueError('Invalid generation options')
        self.eval()
        dev = self.token.weight.device
        ids = torch.tensor([[BOS]+list(prompt.encode('utf-8'))], device=dev)
        g = torch.Generator(device=dev).manual_seed(seed)
        new = []
        for _ in range(tokens):
            logits,_,_ = self(ids[:, -self.context:])
            last = logits[:, -1].clone()
            last[:, BOS] = -torch.inf
            if temperature == 0:
                nxt = last.argmax(-1,keepdim=True)
            else:
                v,i = (last/temperature).topk(top_k)
                nxt = i.gather(-1, torch.multinomial(v.softmax(-1),1,generator=g))
            new.append(int(nxt.item()))
            ids = torch.cat((ids,nxt),dim=1)
        return bytes(new).decode('utf-8',errors='replace')


def load_corpus(path, demo=False):
    if demo:
        # Separate rows, repetitive synthetic mini corpus; never evidence of general ability.
        data = ''.join(f'item {i:05d}: structure -> relation -> information -> feedback.\n'
                       f'{i%19} + {i%7} = {i%19+i%7}.\n' for i in range(2000)).encode()
    else:
        if not path:
            raise ValueError('Supply --text UTF-8 text file or explicit --demo')
        data = Path(path).read_bytes()
    import hashlib
    signature = hashlib.sha256(data).hexdigest()
    cut = int(len(data)*0.9)
    def encode(b): return torch.tensor([BOS]+list(b),dtype=torch.long)
    return encode(data[:cut]), encode(data[cut:]), signature


def lm_batch(data,b,t,g,device):
    if len(data) <= t:
        raise ValueError('Corpus split must be longer than context')
    start = randints(len(data)-t,(b,),g)
    window = data[start[:,None]+torch.arange(t+1)]
    return window[:,:-1].to(device), window[:,1:].to(device)


@torch.no_grad()
def evaluate_lm(model,data,device,b,t,batches,seed):
    model.eval(); g=rng(seed); total=0.; n=0; counts=[0]*model.experts if model.arch=='shared' else []
    sync(device); start=time.perf_counter()
    for _ in range(batches):
        x,y=lm_batch(data,b,t,g,device)
        logits,_,c=model(x)
        total += float(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
        n += y.numel()
        if counts: counts=[a+b for a,b in zip(counts,c)]
    sync(device)
    ce=total/n
    return dict(nll=ce, bits_per_byte=ce/math.log(2), byte_perplexity=math.exp(min(ce,30)),
                evaluated_tokens=n, tokens_per_second=n/max(time.perf_counter()-start,1e-9),
                expert_token_counts=counts)


# ------------------ Reproducible training & inference ------------------
def device_for(s):
    if s=='auto': s='cuda' if torch.cuda.is_available() else 'cpu'
    if s=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    return torch.device(s)


def amp_context(args,device):
    if not args.amp: return contextlib.nullcontext()
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('--amp requires CUDA with BF16 support; omit --amp for FP32')
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16)


def checkpoint(path,kind,config,model,opt,step,g,meta):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    torch.save(dict(version=1,kind=kind,config=config,model=model.state_dict(),optimizer=opt.state_dict(),
        step=step,data_rng=g.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], meta=meta),tmp)
    os.replace(tmp,p)


def read_checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def restore(path,kind,config,model,opt,g,meta):
    if not path: return 0
    ck=read_checkpoint(path)
    if ck['kind']!=kind or ck['config']!=config or ck['meta']!=meta:
        raise ValueError('Resume requires same architecture and training settings/corpus')
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
    g.set_state(ck['data_rng']); torch.set_rng_state(ck['cpu_rng'])
    if torch.cuda.is_available() and len(ck['cuda_rng'])==torch.cuda.device_count():
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['step']


def train(args,kind):
    seed_everything(args.seed,args.threads); device=device_for(args.device)
    g=rng(args.seed+10_000)
    if kind=='executor':
        if args.max_train_len<1 or args.max_train_len>3:
            raise ValueError('v0.1 train length must be 1..3 for fixed evaluation protocol')
        config=dict(arch=args.arch,d=args.d,hidden=args.hidden,experts=args.experts,
                    topk=args.topk,fixed_route=args.fixed_route)
        model=Executor(**config).to(device)
        meta=dict(seed=args.seed,batch=args.batch,lr=args.lr,amp=args.amp,
            max_train_len=args.max_train_len,step_weight=args.step_weight,read_weight=args.read_weight,
            route_weight=args.route_weight,balance_weight=args.balance_weight)
    else:
        config=dict(arch=args.arch,d=args.d,hidden=args.hidden,heads=args.heads,
                    depth=args.depth,experts=args.experts,topk=args.topk,context=args.context)
        model=CoreLM(**config).to(device)
        tr,va,signature=load_corpus(args.text,args.demo)
        if min(len(tr),len(va))<=args.context: raise ValueError('Corpus splits too short')
        meta=dict(seed=args.seed,batch=args.batch,lr=args.lr,amp=args.amp,
                  corpus_sha256=signature,balance_weight=args.balance_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    start=restore(args.resume,kind,config,model,opt,g,meta)
    if args.steps<=start: raise ValueError('--steps is TOTAL steps and must exceed checkpoint step')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    emit(event='start',kind=kind,device=str(device),parameters=nparams(model),config=config,
         start_step=start,meta=meta)
    model.train(); sync(device); clock=time.perf_counter(); acc=0.; unit=0
    for step in range(start+1,args.steps+1):
        opt.zero_grad(set_to_none=True)
        with amp_context(args,device):
            if kind=='executor':
                length=int(randints(args.max_train_len,(1,),g))+1
                batch=make_batch(args.batch,length,g,'train',device)
                loss,_=exec_loss(model,batch,args.step_weight,args.read_weight,args.route_weight,args.balance_weight)
                unit += args.batch
            else:
                x,y=lm_batch(tr,args.batch,args.context,g,device)
                logits,aux,_=model(x)
                loss=F.cross_entropy(logits.flatten(0,1),y.flatten())+args.balance_weight*aux
                unit += y.numel()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step(); acc+=float(loss.detach())
        if step%args.log_every==0 or step==args.steps:
            sync(device); elapsed=time.perf_counter()-clock
            interval=step-max(start,((step-1)//args.log_every)*args.log_every)
            emit(event='train',step=step,loss=acc/interval,units_per_second=unit/max(elapsed,1e-9))
            clock=time.perf_counter();acc=0.;unit=0
        if args.save_every and step%args.save_every==0:
            checkpoint(args.out,kind,config,model,opt,step,g,meta)
    checkpoint(args.out,kind,config,model,opt,args.steps,g,meta)
    if device.type=='cuda':
        emit(event='memory',peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
             peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
    if kind=='executor':
        report=evaluate_executor(model,device,args.eval_batch,args.eval_batches,args.eval_seed)
    else:
        report=evaluate_lm(model,va,device,args.eval_batch,args.context,args.eval_batches,args.eval_seed)
    emit(event='evaluation',kind=kind,step=args.steps,parameters=nparams(model),metrics=report,checkpoint=args.out)


def load_model(path,kind,device):
    ck=read_checkpoint(path)
    if ck['kind']!=kind: raise ValueError('Wrong checkpoint type')
    model=(Executor if kind=='executor' else CoreLM)(**ck['config']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    return model,ck


def predict_executor(model, spec, device):
    """Public JSON IR inference. No oracle or targets are used in this path."""
    if not isinstance(spec,dict) or not isinstance(spec.get('memory'),dict):
        raise ValueError('Expected object with memory, modulus, start_key, steps')
    mod=spec['modulus']
    if type(mod) is not int or not 2<=mod<=Q: raise ValueError('modulus must be integer 2..8')
    pairs=[(int(k),v) for k,v in spec['memory'].items()]
    mem=dict(pairs)
    if not 1<=len(mem)<=N_KEYS or len(mem)!=len(pairs): raise ValueError('Invalid or duplicate memory IDs')
    if any(not 0<=k<N_KEYS or type(v) is not int or not 0<=v<mod for k,v in pairs):
        raise ValueError('Memory keys 0..15 and values 0..modulus-1 required')
    start=spec['start_key']
    if type(start) is not int or start not in mem: raise ValueError('start_key must be present')
    steps=spec['steps']
    if not isinstance(steps,list) or not 1<=len(steps)<=128: raise ValueError('Require 1..128 steps')
    queries=[start]; refs=[]; literals=[]; ops=[]
    for step in steps:
        if step['op'] not in OPS or ('key' in step)==('literal' in step):
            raise ValueError('Each step must specify an op and exactly one key or literal')
        ops.append(OPS.index(step['op'])); ref='key' in step; refs.append(ref)
        val=step['key'] if ref else step['literal']
        if type(val) is not int or (val not in mem if ref else not 0<=val<mod):
            raise ValueError('Invalid operand')
        queries.append(val if ref else start); literals.append(0 if ref else val)
    def row(x): return torch.tensor([x],device=device)
    with torch.no_grad():
        o=model(keys=row(list(mem)),values=row(list(mem.values())),queries=row(queries),
            use_ref=row(refs),literals=row(literals),ops=row(ops),mod=row(mod))
        probs=o['logits'].softmax(-1)[0]; confidence,answers=probs.max(-1)
        resolved=[list(mem)[i] for i in o['attention'][0].argmax(-1).tolist()]
    return dict(final=int(answers[-1]),step_predictions=answers.tolist(),
        step_confidence=confidence.tolist(),retrieved_start_key=resolved[0],
        retrieved_operand_keys=[k if ref else None for k,ref in zip(resolved[1:],refs)],
        expert_token_counts=o['counts'],router_top1=(o['route'][0].argmax(-1).tolist()
            if o['route'] is not None and not model.fixed_route else None))


def selftest():
    seed_everything(313,2); dev=torch.device('cpu')
    g=rng(8); a=make_batch(64,3,g,'train'); b=make_batch(64,3,rng(8),'train')
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert not forbidden(a['ops'][:,:-1],a['ops'][:,1:]).any()
    p=make_batch(64,3,rng(9),'pair')
    assert forbidden(p['ops'][:,:-1],p['ops'][:,1:]).any(1).all()
    m=make_batch(64,3,rng(9),'mod')['mod']; assert set(m.tolist()) <= set(NEW_MODS)
    # Independently check each target against Python integer arithmetic.
    for i in range(64):
        cap=int(a['mod'][i]); values=dict(zip(a['keys'][i].tolist(),a['values'][i].tolist()))
        s=values[int(a['queries'][i,0])]
        for j in range(3):
            v=values[int(a['queries'][i,j+1])] if a['use_ref'][i,j] else int(a['literals'][i,j])
            op=int(a['ops'][i,j]); s=((s+v)%cap,(s-v)%cap,(s*v)%cap,max(s,v))[op]
            assert s==int(a['labels'][i,j])
    for arch in ('shared','dense','gru'):
        net=Executor(arch=arch,d=32,hidden=48)
        loss,o=exec_loss(net,a); loss.backward()
        assert torch.isfinite(loss) and net.project[0].weight.grad is not None
        assert o['logits'].shape==(64,3,Q)
        if arch=='shared': assert sum(o['counts'])==64*3
    # Sparse dispatch equals direct evaluation for a known fixed assignment.
    bank=SparseBank(8,12,5,4,1); x=torch.randn(9,8); fixed=torch.arange(9)%4
    out,_,_,counts=bank(x,fixed_ids=fixed)
    direct=torch.stack([bank.bank[int(i)](z[None])[0] for z,i in zip(x,fixed)])
    torch.testing.assert_close(out,direct); assert sum(counts)==9
    # Confirm selected expert only, with nonzero router gradient through top1 STE.
    bank=SparseBank(8,12,5,4,1)
    with torch.no_grad(): bank.router.weight.zero_(); bank.router.bias.copy_(torch.tensor([5.,0.,0.,0.]))
    y,_,_,counts=bank(x); y.square().mean().backward()
    assert counts==[9,0,0,0] and bank.bank[1].net[0].weight.grad is None
    assert bank.router.weight.grad is not None and bank.router.weight.grad.abs().sum()>0
    for arch in ('shared','shared_dense','dense'):
        lm=CoreLM(arch=arch,d=32,hidden=48,heads=4,depth=2,context=16)
        ids=torch.randint(BYTE_VOCAB,(2,12)); altered=ids.clone();altered[:,7:]=(altered[:,7:]+1)%BYTE_VOCAB
        y,aux,c=lm(ids); yy,_,_=lm(altered)
        torch.testing.assert_close(y[:,:7],yy[:,:7],atol=1e-5,rtol=1e-5)
        assert y.shape==(2,12,BYTE_VOCAB)
        (y.square().mean()+0.01*aux).backward()
        if arch=='shared': assert sum(c)==2*12*2*2
    emit(event='selftest',status='PASS',checks=['data_reproducibility','heldout_splits','arithmetic_labels',
        'executor_gradients','sparse_dispatch','inactive_expert','router_gradient','causal_lm','lm_gradients'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    def common(s,kind):
        s.add_argument('--arch',choices=('shared','dense','gru') if kind=='executor' else ('shared','shared_dense','dense'),default='shared')
        s.add_argument('--d',type=int,default=64 if kind=='executor' else 128)
        s.add_argument('--hidden',type=int,default=96 if kind=='executor' else 256)
        s.add_argument('--experts',type=int,default=4);s.add_argument('--topk',type=int,default=1 if kind=='executor' else 2)
        s.add_argument('--batch',type=int,default=64 if kind=='executor' else 8)
        s.add_argument('--steps',type=int,default=2000 if kind=='executor' else 1000)
        s.add_argument('--lr',type=float,default=0.002 if kind=='executor' else 0.001)
        s.add_argument('--seed',type=int,default=313);s.add_argument('--threads',type=int,default=4)
        s.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
        s.add_argument('--amp',action='store_true');s.add_argument('--balance-weight',type=float,default=0.01)
        s.add_argument('--eval-batch',type=int,default=128 if kind=='executor' else 8)
        s.add_argument('--eval-batches',type=int,default=20);s.add_argument('--eval-seed',type=int,default=271828)
        s.add_argument('--log-every',type=int,default=100);s.add_argument('--save-every',type=int,default=500)
        s.add_argument('--out',default='runs/'+kind+'.pt');s.add_argument('--resume')
    s=sub.add_parser('train-exec');common(s,'executor')
    s.add_argument('--fixed-route',action='store_true');s.add_argument('--max-train-len',type=int,default=3)
    s.add_argument('--step-weight',type=float,default=0.5);s.add_argument('--read-weight',type=float,default=0.1)
    s.add_argument('--route-weight',type=float,default=0.1)
    s=sub.add_parser('train-lm');common(s,'lm')
    s.add_argument('--heads',type=int,default=4);s.add_argument('--depth',type=int,default=4)
    s.add_argument('--context',type=int,default=128);s.add_argument('--text');s.add_argument('--demo',action='store_true')
    s=sub.add_parser('eval-exec');s.add_argument('checkpoint');s.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    s.add_argument('--batch',type=int,default=128);s.add_argument('--batches',type=int,default=50);s.add_argument('--seed',type=int,default=271828)
    s=sub.add_parser('generate');s.add_argument('checkpoint');s.add_argument('--prompt',default='structure');s.add_argument('--tokens',type=int,default=100)
    s.add_argument('--temperature',type=float,default=0.8);s.add_argument('--top-k',type=int,default=40);s.add_argument('--seed',type=int,default=313)
    s.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    s=sub.add_parser('predict-exec');s.add_argument('checkpoint');s.add_argument('input_json')
    s.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');s.add_argument('--seed',type=int,default=313)
    s=sub.add_parser('eval-lm');s.add_argument('checkpoint');s.add_argument('--text');s.add_argument('--demo',action='store_true')
    s.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');s.add_argument('--batch',type=int,default=8)
    s.add_argument('--batches',type=int,default=50);s.add_argument('--seed',type=int,default=271828)
    sub.add_parser('selftest');args=p.parse_args()
    if args.cmd=='selftest': selftest();return
    if args.cmd.startswith('train'):
        if min(args.batch,args.steps,args.log_every,args.eval_batch,args.eval_batches,args.d,args.hidden,args.threads)<1 or args.save_every<0:
            p.error('Batch, steps, dimensions and intervals must be positive (save_every may be 0)')
        train(args,'executor' if args.cmd=='train-exec' else 'lm');return
    seed_everything(args.seed);dev=device_for(args.device)
    if args.cmd=='eval-exec':
        model,_=load_model(args.checkpoint,'executor',dev)
        emit(event='evaluation',parameters=nparams(model),metrics=evaluate_executor(model,dev,args.batch,args.batches,args.seed))
    elif args.cmd=='predict-exec':
        model,_=load_model(args.checkpoint,'executor',dev)
        spec=json.loads(Path(args.input_json).read_text(encoding='utf-8'))
        emit(event='prediction',**predict_executor(model,spec,dev))
    elif args.cmd=='eval-lm':
        model,ck=load_model(args.checkpoint,'lm',dev)
        _,va,signature=load_corpus(args.text,args.demo)
        if signature!=ck['meta']['corpus_sha256']:
            raise ValueError('eval-lm expects the original corpus (same SHA256)')
        emit(event='evaluation',parameters=nparams(model),metrics=evaluate_lm(model,va,dev,args.batch,model.context,args.batches,args.seed))
    else:
        model,_=load_model(args.checkpoint,'lm',dev)
        print(args.prompt+model.generate(args.prompt,args.tokens,args.temperature,args.top_k,args.seed))


if __name__=='__main__': main()

