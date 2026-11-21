"""Determinism helpers, metrics serialization and device-independent timing."""
from __future__ import annotations
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
import torch


def select_device(requested: str) -> torch.device:
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    if requested not in ('cpu','cuda'):
        raise ValueError(f'Invalid device: {requested}')
    return torch.device(requested)


def seed_all(seed: int, threads: int = 2) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()


def digest_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),encoding='utf-8')
    os.replace(tmp, path)


def append_jsonl(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as file:
        file.write(json.dumps(obj,ensure_ascii=False,default=str)+'\n')


def environment_info(device: torch.device) -> dict:
    try:
        git_sha = subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,timeout=2).decode().strip()
    except Exception:
        git_sha = None
    return {
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'git_commit': git_sha, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    }
