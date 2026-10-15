"""Experiment B: complete GPT-2 transformer-layer ablation on WikiText-2.

Run: venv/bin/python ablation2.py
Uses Experiment A's model, data, seed, context, batching, and evaluation defaults.
Plots and results default to ab2_plots/. Indices are zero-based.

skip bypasses the entire block. zero_contribution executes the block but cancels
its net residual update (output - input); for pre-normalized GPT-2 this equals
skip mathematically. zero_output zeros the full block output, including residual
information, providing a distinct control. Final model normalization remains.
Overall next-token metrics cannot establish lexical/syntactic specialization;
that hypothesis requires additional targeted linguistic evaluations.
"""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablation1 import PERCENTAGES, evaluate, token_blocks
from model import load_model

ROOT = Path(__file__).resolve().parent
MODES = ('skip', 'zero_contribution', 'zero_output')


class LayerInterventions:
    """Observe output structure, then temporarily replace selected forwards.

    Both tensor-returning and older tuple-returning GPT-2 blocks are supported.
    Cache and attention recording are disabled for this evaluation.
    """
    def __init__(self, backbone):
        self.blocks = backbone.h
        self.templates = {}
        self.handles = []
        for index, block in enumerate(self.blocks):
            def observe(module, inputs, output, index=index):
                # Keep only structure, never baseline activation tensors.
                self.templates[index] = None if torch.is_tensor(output) else tuple(
                    None for _ in output[1:])
            self.handles.append(block.register_forward_hook(observe))

    @contextmanager
    def apply(self, layers, mode):
        if mode not in MODES:
            raise ValueError(f'Unknown intervention: {mode}')
        saved = []
        try:
            for index in layers:
                block = self.blocks[index]
                original = block.forward
                had_override = 'forward' in block.__dict__
                template = self.templates[index]
                saved.append((block, original, had_override))

                def forward(hidden_states, *args, original=original,
                            template=template, **kwargs):
                    if mode == 'skip':
                        return hidden_states if template is None else (hidden_states, *template)
                    output = original(hidden_states, *args, **kwargs)
                    value = output if torch.is_tensor(output) else output[0]
                    # Cancel the net block update exactly, preserving residual input.
                    replacement = hidden_states if mode == 'zero_contribution' else torch.zeros_like(value)
                    return replacement if torch.is_tensor(output) else (replacement, *output[1:])
                block.forward = forward
            yield
        finally:
            for block, original, had_override in reversed(saved):
                if had_override:
                    block.forward = original
                else:
                    del block.forward

    def close(self):
        for handle in self.handles:
            handle.remove()


def is_contiguous(layers):
    return all(b == a + 1 for a, b in zip(layers, layers[1:]))


def conditions(layers, trials, seed, group_size):
    """Equal layer budgets; contiguous windows anchored early/middle/late."""
    n = len(layers)
    for layer in layers:
        yield 'single_layer', 0, 100 / n, [layer]
    counts = sorted({min(n, group_size), *(max(1, round(n * p / 100)) for p in PERCENTAGES)})
    for count in counts:
        windows = [layers[start:start + count] for start in range(n - count + 1)
                   if is_contiguous(layers[start:start + count])]
        if windows:
            for name, index in [('early', 0), ('middle', (len(windows) - 1) // 2),
                                ('late', len(windows) - 1)]:
                yield f'contiguous_{name}', 0, 100 * count / n, windows[index]
        # A single layer or full removal cannot be noncontiguous on a full stack.
        if count < 2 or count == n:
            continue
        for trial in range(trials):
            rng = np.random.default_rng(seed + trial + count * 1000)
            for _ in range(1000):
                selected = sorted(int(x) for x in rng.choice(layers, count, replace=False))
                if not is_contiguous(selected):
                    yield 'noncontiguous_random', trial, 100 * count / n, selected
                    break
            else:
                raise RuntimeError('Could not generate a noncontiguous layer subset.')


def save_plots(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    baseline = frame[frame.strategy == 'baseline'].iloc[0]
    for metric, label in [('accuracy', 'Next-token accuracy (%)'), ('perplexity', 'Perplexity')]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, mode in zip(axes, MODES):
            for strategy, subset in frame[(frame['mode'] == mode) & (frame.strategy != 'single_layer')].groupby('strategy', sort=False):
                summary = subset.groupby('disabled_layers')[metric].agg(['mean', 'std'])
                ax.errorbar(summary.index, summary['mean'], yerr=summary['std'].fillna(0),
                            marker='o', capsize=3, label=strategy.replace('_', ' '))
            ax.axhline(baseline[metric], color='black', linestyle='--', label='baseline')
            ax.set(xlabel='Removed layers', ylabel=label, title=mode.replace('_', ' '))
            if metric == 'perplexity':
                ax.set_yscale('log')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle('GPT-2 transformer-layer ablation: increasing group size')
        fig.tight_layout()
        fig.savefig(output / f'{metric}_group_ablation.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode in MODES:
            subset = frame[(frame['mode'] == mode) & (frame.strategy == 'single_layer')]
            ax.plot(subset.layer_index, subset[metric], marker='o', label=mode.replace('_', ' '))
        ax.axhline(baseline[metric], color='black', linestyle='--', label='baseline')
        ax.set(xlabel='Removed layer index (early → late)', ylabel=label,
               title='GPT-2 individual transformer-layer ablation')
        if metric == 'perplexity':
            ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}_single_layer.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/wikitext2/wikitext-2-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ab2_plots')
    parser.add_argument('--layers', type=int, nargs='+', help='Zero-based layer indices')
    parser.add_argument('--eval-blocks', type=int, default=16)
    parser.add_argument('--calibration-blocks', type=int, default=8,
                        help='Retained from Experiment A; no importance calibration is needed here')
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=3, help='Additional layer-removal budget')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.eval_blocks, args.calibration_blocks, args.batch_size, args.trials,
           args.group_size) < 1 or args.block_size < 2:
        parser.error('Counts must be positive and block-size must be at least 2.')
    torch.manual_seed(args.seed)
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
    test = token_blocks(model.tokenizer, args.data_dir / 'test-00000-of-00001.parquet',
                        args.block_size, args.eval_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    interventions = LayerInterventions(backbone)
    rows = []
    try:
        baseline = evaluate(backbone, test, args.batch_size)
        rows.append(dict(mode='none', strategy='baseline', trial=0, requested_percent=0,
                         actual_percent=0, disabled_layers=0, layers='[]', layer_index=-1, **baseline))
        print(f'Baseline: {baseline}', flush=True)
        # Duplicate budgets are evaluated once; paired modes use exactly the same subsets.
        cache = {}
        for strategy, trial, percent, selected in conditions(layers, args.trials, args.seed, args.group_size):
            for mode in MODES:
                key = (mode, tuple(selected))
                if key not in cache:
                    with interventions.apply(selected, mode):
                        cache[key] = evaluate(backbone, test, args.batch_size)
                metrics = cache[key]
                rows.append(dict(mode=mode, strategy=strategy, trial=trial,
                                 requested_percent=percent, actual_percent=100 * len(selected) / len(layers),
                                 disabled_layers=len(selected), layers=json.dumps(selected),
                                 layer_index=selected[0] if strategy == 'single_layer' else -1, **metrics))
                print(f'{mode} {strategy} trial={trial + 1} layers={selected} '
                      f"accuracy={metrics['accuracy']:.4f}% perplexity={metrics['perplexity']:.3f}", flush=True)
    finally:
        interventions.close()
    with (args.output_dir / 'ablation_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {**vars(args), 'data_dir': str(args.data_dir), 'output_dir': str(args.output_dir),
                'layers': layers, 'model_type': backbone.config.model_type,
                'test_tokens': baseline['tokens'], 'percentage_budgets': PERCENTAGES,
                'calibration_used': False,
                'interventions': {'skip': 'identity; block is not executed',
                                  'zero_contribution': 'execute block and cancel net residual update; equivalent to skip for GPT-2',
                                  'zero_output': 'zero entire block output, including residual stream'},
                'group_policy': 'early/middle/late contiguous windows; random subsets conditioned on not being one contiguous run; paired subsets across modes',
                'limitations': 'Overall language modeling metrics do not isolate lexical, syntactic, or contextual abilities.'}
    (args.output_dir / 'experiment_config.json').write_text(json.dumps(metadata, indent=2))
    save_plots(rows, args.output_dir)
    print(f'Results and plots saved to {args.output_dir}')


if __name__ == '__main__':
    main()
