"""Shared offline GPT-2 evaluation utilities for the ParaNMT experiments."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nai3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
COLUMNS = ['original_text', 'paraphrased_sentence',
           'grammatical_but_semantically_unusual', 'syntactically_incorrect']

def parser(description, folder):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data', type=Path, default=ROOT/'dataset/para_nmt_first_1000_augmented.csv')
    p.add_argument('--model', type=Path, default=ROOT/'models/gpt2')
    p.add_argument('--output', type=Path, default=ROOT/folder)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--limit', type=int, help='Evaluate only the first N rows (default: all).')
    p.add_argument('--seed', type=int, default=42)
    return p

def setup(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    frame = pd.read_csv(args.data)
    missing = set(COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if args.limit is not None:
        if args.limit < 1: raise ValueError('--limit must be positive')
        frame = frame.iloc[:args.limit]
    if frame.empty or frame[COLUMNS].isna().any().any() or frame[COLUMNS].map(lambda x: not isinstance(x, str) or not x.strip()).any().any():
        raise ValueError('All four columns must contain nonempty sentences in every selected row.')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, attn_implementation='eager').to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(device=device, rows=len(frame), columns=COLUMNS, torch_version=torch.__version__)
    (args.output/f'{getattr(args, "experiment", "run")}_config.json').write_text(json.dumps(config, indent=2))
    return frame, tokenizer, model

def encode(tokenizer, model, sentence, bos=False):
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if bos: ids = [tokenizer.eos_token_id]+ids
    if len(ids) > model.config.n_positions:
        raise ValueError(f'Sentence has {len(ids)} tokens; context limit is {model.config.n_positions}. No silent truncation.')
    return torch.tensor([ids], device=next(model.parameters()).device)

def save_matrix(matrix, labels, title, path, xlabel='', ylabel='', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*.32), max(6, matrix.shape[0]*.3)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    if matrix.shape[0] == len(labels):
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

@torch.inference_mode()
def representations(frame, tokenizer, model):
    """Return [row, variant, layer (embedding=0), pooling, hidden]."""
    result=[]
    for idx, row in frame.iterrows():
        variants=[]
        for col in COLUMNS:
            states=model(encode(tokenizer, model, row[col]), output_hidden_states=True, use_cache=False).hidden_states
            variants.append(torch.stack([torch.stack([s[0].mean(0), s[0,-1], s[0].max(0).values]) for s in states]).cpu())
        result.append(torch.stack(variants))
        if len(result)%50==0: print(f'Encoded {len(result)}/{len(frame)} rows', flush=True)
    return torch.stack(result)

def meaning_report(embeddings, frame, output, name, conditions):
    """Each condition is [rows, four variants, hidden]; cyclic negatives never match rows."""
    records=[]
    for label, vectors in conditions:
        vectors=F.normalize(vectors.float(), dim=-1)
        for variant in range(1,4):
            paired=(vectors[:,0]*vectors[:,variant]).sum(-1)
            negative=(vectors[:,0]*vectors[:,variant].roll(1,0)).sum(-1) if len(frame)>1 else torch.full_like(paired, float('nan'))
            for i, row in enumerate(frame.index):
                records.append(dict(row_index=int(row), condition=label, comparison=COLUMNS[variant], cosine=float(paired[i]), shuffled_cosine=float(negative[i]), paired_minus_shuffled=float(paired[i]-negative[i])))
    df=pd.DataFrame(records); df.to_csv(output/f'{name}_pairs.csv', index=False)
    summary=df.groupby(['condition','comparison'], sort=False)[['cosine','shuffled_cosine','paired_minus_shuffled']].mean().reset_index()
    summary.to_csv(output/f'{name}_summary.csv',index=False)
    for metric in ['cosine','paired_minus_shuffled']:
        pivot=summary.pivot(index='condition', columns='comparison', values=metric).reindex(index=[c[0] for c in conditions])
        ax=pivot.plot(marker='o',figsize=(12,6), title=f'{name}: {metric} (mean across rows)')
        ax.set_ylabel(metric); ax.tick_params(axis='x',rotation=35)
        plt.tight_layout(); plt.savefig(output/f'{name}_{metric}.png',dpi=160); plt.close()
    return df
