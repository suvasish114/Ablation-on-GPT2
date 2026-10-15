"""GPT-2 attention-head ablation on held-out WikiText-2 next-token prediction.

Run: venv/bin/python ablition.py
Selected layers: venv/bin/python ablition.py --layers 0 1 2
Larger evaluation: venv/bin/python ablition.py --eval-blocks 128 --calibration-blocks 32
Percentages refer to heads in --layers (all layers by default). Raw text is kept
because punctuation and case are part of GPT-2's language-modeling task.
This evaluates overall language modeling, not individual linguistic relationships.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import load_model

ROOT = Path(__file__).resolve().parent
PERCENTAGES = (5, 10, 20, 30, 50, 75)


def token_blocks(tokenizer, path, block_size, max_blocks):
    """Use fixed, disjoint contexts; score every position except the first."""
    frame = pd.read_parquet(path, columns=['text'])
    tokens = []
    target = block_size * max_blocks
    for text in frame['text']:
        if not isinstance(text, str) or not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= target:
            break
    count = min(len(tokens) // block_size, max_blocks)
    if count == 0:
        raise ValueError(f'Not enough text in {path} for a complete block.')
    return torch.tensor(tokens[:count * block_size]).reshape(count, block_size)


class HeadGates:
    """Zero each head's output before attention's output projection.

    This avoids backend-dependent head_mask support and leaves weights intact.
    """
    def __init__(self, backbone):
        self.handles = []
        self.gates = []
        for block in backbone.h:
            heads = block.attn.num_heads
            gate = torch.ones(heads, device=next(backbone.parameters()).device,
                              requires_grad=True)
            self.gates.append(gate)

            def mask(module, inputs, gate=gate):
                value = inputs[0]
                shaped = value.reshape(*value.shape[:-1], gate.numel(), -1)
                output = (shaped * gate[:, None]).reshape_as(value)
                return (output, *inputs[1:])

            self.handles.append(block.attn.c_proj.register_forward_pre_hook(mask))

    def set_disabled(self, heads):
        with torch.no_grad():
            for gate in self.gates:
                gate.fill_(1)
            for layer, head in heads:
                self.gates[layer][head] = 0

    def close(self):
        for handle in self.handles:
            handle.remove()


def logits_for(backbone, inputs):
    hidden = backbone(input_ids=inputs, use_cache=False).last_hidden_state
    # GPT-2 uses its input embedding weights as its language-model output head.
    return F.linear(hidden[:, :-1], backbone.get_input_embeddings().weight)


@torch.no_grad()
def evaluate(backbone, blocks, batch_size):
    correct = total = 0
    loss_sum = 0.0
    device = next(backbone.parameters()).device
    for batch in blocks.split(batch_size):
        batch = batch.to(device)
        logits = logits_for(backbone, batch)
        labels = batch[:, 1:]
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    labels.reshape(-1), reduction='sum').item()
    loss = loss_sum / total
    return {'accuracy': 100 * correct / total, 'loss': loss,
            'perplexity': math.exp(min(loss, 700)), 'tokens': total}


def importance_scores(backbone, gates, blocks, batch_size):
    """Token-weighted mean absolute gradient of loss w.r.t. each head gate."""
    scores = [torch.zeros_like(gate) for gate in gates.gates]
    total = 0
    device = next(backbone.parameters()).device
    for batch in blocks.split(batch_size):
        for gate in gates.gates:
            gate.grad = None
        batch = batch.to(device)
        logits = logits_for(backbone, batch)
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        for score, gate in zip(scores, gates.gates):
            if gate.grad is None:
                raise RuntimeError('Head gate was not connected to the loss.')
            score += gate.grad.detach().abs() * labels.numel()
        total += labels.numel()
    return torch.stack(scores).detach().cpu().numpy() / total


def random_order(candidates, rng, group_size=None):
    if group_size is None:
        return [candidates[i] for i in rng.permutation(len(candidates))]
    groups = []
    for layer in sorted({layer for layer, _ in candidates}):
        heads = [head for lyr, head in candidates if lyr == layer]
        rng.shuffle(heads)
        groups.extend([[(layer, head) for head in heads[i:i + group_size]]
                       for i in range(0, len(heads), group_size)])
    return [head for i in rng.permutation(len(groups)) for head in groups[i]]


def save_plots(rows, scores, layers, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    for metric, label in [('accuracy', 'Next-token accuracy (%)'),
                          ('perplexity', 'Perplexity')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, subset in frame.groupby('strategy', sort=False):
            summary = subset.groupby('actual_percent')[metric].agg(['mean', 'std'])
            ax.errorbar(summary.index, summary['mean'],
                        yerr=summary['std'].fillna(0), marker='o', capsize=3,
                        label=strategy.replace('_', ' '))
        ax.set(xlabel='Disabled heads in selected layers (%)', ylabel=label,
               title='GPT-2 attention-head ablation')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}_ablation.png', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * .35)))
    heatmap = ax.imshow(scores[layers], aspect='auto', cmap='viridis')
    ax.set(xlabel='Head index', ylabel='Layer index', title='Validation head importance')
    ax.set_yticks(range(len(layers)), labels=layers)
    fig.colorbar(heatmap, ax=ax, label='Mean absolute loss gradient')
    fig.tight_layout()
    fig.savefig(output / 'head_importance.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/wikitext2/wikitext-2-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'plots')
    parser.add_argument('--layers', type=int, nargs='+', help='Zero-based layer indices')
    parser.add_argument('--eval-blocks', type=int, default=16)
    parser.add_argument('--calibration-blocks', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.eval_blocks, args.calibration_blocks, args.batch_size,
           args.trials, args.group_size) < 1 or args.block_size < 2:
        parser.error('Counts must be positive and block-size must be at least 2.')
    torch.manual_seed(args.seed)
    # Avoid excessive CPU thread overhead on small inference batches.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_model()
    backbone = model[0].auto_model
    if backbone.config.model_type != 'gpt2' or not hasattr(backbone, 'h'):
        raise ValueError('This experiment supports the configured GPT-2 backbone only.')
    if args.block_size > backbone.config.n_positions:
        parser.error('block-size exceeds the model context length.')
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    layers = sorted(set(args.layers if args.layers is not None else range(len(backbone.h))))
    if not layers or min(layers) < 0 or max(layers) >= len(backbone.h):
        parser.error('Invalid layer index.')
    candidates = [(layer, head) for layer in layers
                  for head in range(backbone.h[layer].attn.num_heads)]
    test = token_blocks(model.tokenizer, args.data_dir / 'test-00000-of-00001.parquet',
                        args.block_size, args.eval_blocks)
    calibration = token_blocks(model.tokenizer,
                               args.data_dir / 'validation-00000-of-00001.parquet',
                               args.block_size, args.calibration_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gates = HeadGates(backbone)
    rows = []
    try:
        baseline = evaluate(backbone, test, args.batch_size)
        print(f"Baseline accuracy: {baseline['accuracy']:.4f}% ({baseline['tokens']} tokens)", flush=True)
        scores = importance_scores(backbone, gates, calibration, args.batch_size)
        ranked = sorted(candidates, key=lambda item: (scores[item], item))
        orders = [('least_important', 0, ranked)]
        for trial in range(args.trials):
            orders.append(('random_individual', trial,
                           random_order(candidates, np.random.default_rng(args.seed + trial))))
            orders.append(('random_groups', trial,
                           random_order(candidates, np.random.default_rng(args.seed + trial),
                                        args.group_size)))
        for strategy, trial, order in orders:
            rows.append(dict(strategy=strategy, trial=trial, requested_percent=0,
                             actual_percent=0, disabled_heads=0, heads='[]', **baseline))
            for percent in PERCENTAGES:
                count = max(1, round(len(candidates) * percent / 100))
                disabled = order[:count]
                # Groups are disabled together except a partial final group needed
                # to match the same head budget as the other strategies.
                gates.set_disabled(disabled)
                metrics = evaluate(backbone, test, args.batch_size)
                rows.append(dict(strategy=strategy, trial=trial, requested_percent=percent,
                                 actual_percent=100 * count / len(candidates),
                                 disabled_heads=count, heads=json.dumps(disabled), **metrics))
                print(f"{strategy} trial={trial + 1} requested={percent}% "
                      f"heads={count}/{len(candidates)} accuracy={metrics['accuracy']:.4f}% "
                      f"perplexity={metrics['perplexity']:.3f}", flush=True)
    finally:
        gates.close()
    with (args.output_dir / 'ablation_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.output_dir / 'head_importance.npy', scores)
    metadata = {**vars(args), 'data_dir': str(args.data_dir), 'output_dir': str(args.output_dir),
                'layers': layers, 'model_type': backbone.config.model_type,
                'test_tokens': baseline['tokens'], 'calibration_tokens': calibration.shape[0] * (args.block_size - 1),
                'importance': 'token-weighted mean absolute gradient of validation loss w.r.t. head gates',
                'group_policy': 'random groups within layers; final group may be partial to match head budget'}
    (args.output_dir / 'experiment_config.json').write_text(json.dumps(metadata, indent=2))
    save_plots(rows, scores, layers, args.output_dir)
    summary = pd.DataFrame(rows).groupby(['strategy', 'requested_percent'])['accuracy'].agg(['mean', 'std'])
    print('\nAccuracy summary (mean and standard deviation in percentage points):')
    print(summary.fillna(0).to_string(float_format=lambda value: f'{value:.4f}'))
    print(f'\nResults and plots saved to {args.output_dir}')


if __name__ == '__main__':
    main()
