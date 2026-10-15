"""Experiment E: GPT-2 attention-pattern ablation on held-out WikiText-2.

Run: venv/bin/python ablation5.py
Selected layers: venv/bin/python ablation5.py --layers 0 1 2
Local window: venv/bin/python ablation5.py --local-window 8
Defaults match ablation1.py; plots/results are saved in ab5_plots/.

Uniform attention: replace selected heads' learned distributions with equal
weights over causally available keys. Future tokens remain inaccessible.
Local attention: mask distant keys before softmax, retaining learned weights
and renormalizing over the causal window. Window counts include the current
position: window=1 is self-only, window=context_length is an intact control.

Studies: increasingly many affected heads (same budgets/random trials/grouping
as Experiment A, plus most-important-first); individual-layer interventions;
early/middle/late/all-layer uniform controls and local-window sweeps. Importance
is calibrated on validation data using Experiment A's head-gate gradients.
These rankings estimate sensitivity to head removal, not necessarily to pattern
replacement. Test data is used only for next-token accuracy/loss/perplexity.

Weights, value projections, output projections, residuals, normalization, token
embeddings, and positions are preserved. Uses explicit eager GPT-2 attention to
intervene on distributions independently of optimized attention backends.
The implementation supports the installed GPT-2 two-output attention interface,
without cross-attention, caching, training dropout, or reordered/upcast attention.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablation1 import (PERCENTAGES, HeadGates, evaluate, importance_scores,
                       random_order, token_blocks)
from model import load_model

ROOT = Path(__file__).resolve().parent


def attention_distributions(query, key, attention_mask, scaling, mode, window):
    """Return learned and modified distributions with explicit causal masking.

    This experiment uses complete, unpadded contexts with no key/value cache.
    Additive masks retain their biases for learned/local attention; uniform
    attention retains permitted connections only.
    """
    length = query.size(-2)
    if key.size(-2) != length:
        raise ValueError('Cached or unequal-length attention is not supported.')
    positions = torch.arange(length, device=query.device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0)[None, None, :, :]
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            allowed = allowed & attention_mask
        else:
            if not attention_mask.is_floating_point():
                raise ValueError('Expected a floating-point additive attention mask.')
            allowed = allowed & (attention_mask > torch.finfo(attention_mask.dtype).min / 2)
            scores = scores + attention_mask
    if not allowed.any(-1).all():
        raise ValueError('An attention query has no available keys.')
    minimum = torch.finfo(scores.dtype).min
    learned = torch.softmax(scores.masked_fill(~allowed, minimum), dim=-1)
    if mode == 'uniform':
        modified = allowed.to(dtype=scores.dtype)
        modified = modified / modified.sum(-1, keepdim=True)
    elif mode == 'local':
        if window < 1:
            raise ValueError('Local window must be positive.')
        local_allowed = allowed & (distance < window)[None, None, :, :]
        if not local_allowed.any(-1).all():
            raise ValueError('A local-attention query has no available keys.')
        modified = torch.softmax(scores.masked_fill(~local_allowed, minimum), dim=-1)
    else:
        raise ValueError(f'Unknown attention pattern: {mode}')
    return learned, modified


class AttentionPatterns:
    """Replace attention forwards temporarily; leave model parameters intact."""
    def __init__(self, backbone):
        self.backbone = backbone
        self.mode = None
        self.window = 8
        self.masks = {}
        self.saved = []
        try:
            for layer, block in enumerate(backbone.h):
                module = block.attn
                if module.is_cross_attention or module.reorder_and_upcast_attn:
                    raise ValueError('Cross-attention and reordered/upcast attention are not supported.')
                original = module.forward
                self.saved.append((module, original, 'forward' in module.__dict__))

                def forward(hidden_states, past_key_values=None, attention_mask=None,
                            encoder_hidden_states=None, encoder_attention_mask=None,
                            output_attentions=False, module=module, layer=layer,
                            original=original, **kwargs):
                    if self.mode is None or layer not in self.masks:
                        return original(hidden_states, past_key_values=past_key_values,
                                        attention_mask=attention_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_attentions=output_attentions, **kwargs)
                    if module.training or past_key_values is not None or kwargs.get('use_cache', False):
                        raise ValueError('Interventions require evaluation mode and use_cache=False.')
                    if encoder_hidden_states is not None:
                        raise ValueError('Cross-attention is not supported.')
                    query, key, value = module.c_attn(hidden_states).split(module.split_size, dim=2)
                    shape = (*query.shape[:-1], module.num_heads, module.head_dim)
                    query = query.view(shape).transpose(1, 2)
                    key = key.view(shape).transpose(1, 2)
                    value = value.view(shape).transpose(1, 2)
                    learned, modified = attention_distributions(query, key, attention_mask,
                                                               module.scaling, self.mode, self.window)
                    weights = torch.where(self.masks[layer], modified, learned).to(value.dtype)
                    output = torch.matmul(weights, value).transpose(1, 2)
                    output = output.reshape(*output.shape[:-2], module.embed_dim).contiguous()
                    output = module.resid_dropout(module.c_proj(output))
                    return output, weights
                module.forward = forward
        except Exception:
            self.close()
            raise

    def set(self, mode=None, heads=(), window=8):
        if mode not in (None, 'uniform', 'local'):
            raise ValueError(f'Unknown attention pattern: {mode}')
        if window < 1:
            raise ValueError('Local window must be positive.')
        self.mode = mode
        self.window = window
        self.masks = {}
        device = next(self.backbone.parameters()).device
        for layer, head in heads:
            if layer < 0 or layer >= len(self.backbone.h):
                raise ValueError('Invalid layer index.')
            count = self.backbone.h[layer].attn.num_heads
            if head < 0 or head >= count:
                raise ValueError('Invalid head index.')
            if layer not in self.masks:
                self.masks[layer] = torch.zeros(1, count, 1, 1, dtype=torch.bool, device=device)
            self.masks[layer][0, head, 0, 0] = True

    def close(self):
        for module, original, had_override in reversed(self.saved):
            if had_override:
                module.forward = original
            else:
                del module.forward
        self.saved.clear()
        self.masks.clear()
        self.mode = None


@torch.no_grad()
def capture_examples(backbone, patterns, calibration, layer, head, local_window):
    examples = {}
    captured = []
    def observe(module, args, output):
        captured.append(output[1][0, head].detach().cpu().numpy())
    handle = backbone.h[layer].attn.register_forward_hook(observe)
    device = next(backbone.parameters()).device
    try:
        for name, mode, window in [('learned', None, local_window),
                                   ('uniform', 'uniform', local_window),
                                   ('local', 'local', local_window),
                                   ('self_only', 'local', 1)]:
            patterns.set(mode, [(layer, head)], window)
            captured.clear()
            backbone(input_ids=calibration[:1].to(device), use_cache=False)
            if len(captured) != 1:
                raise RuntimeError('Could not capture the example attention distribution.')
            examples[name] = captured[0]
    finally:
        handle.remove()
        patterns.set()
    return examples


def save_plots(rows, scores, layers, examples, example_layer, example_head, local_window, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    baseline = frame[frame.study == 'baseline'].iloc[0]
    for metric, label in [('accuracy', 'Next-token accuracy (%)'), ('perplexity', 'Perplexity')]:
        for study, x, xlabel in [('head_budget', 'actual_percent', 'Affected heads in selected layers (%)'),
                                 ('single_layer', 'layer_index', 'Layer index (all heads affected)')]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
            for ax, mode in zip(axes, ('uniform', 'local')):
                subset = frame[(frame.study == study) & (frame['mode'] == mode)]
                for strategy, values in subset.groupby('strategy', sort=False):
                    values = values.drop_duplicates(subset=['trial', x])
                    summary = values.groupby(x)[metric].agg(['mean', 'std'])
                    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'].fillna(0),
                                marker='o', capsize=3, label=strategy.replace('_', ' '))
                ax.axhline(baseline[metric], color='black', linestyle='--', label='baseline')
                ax.set(xlabel=xlabel, ylabel=label,
                       title='Uniform causal attention' if mode == 'uniform' else f'Local causal attention (window={local_window})')
                if metric == 'perplexity':
                    ax.set_yscale('log')
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            fig.suptitle(f'GPT-2 attention-pattern ablation: {study.replace("_", " ")}')
            fig.tight_layout()
            suffix = 'ablation' if study == 'head_budget' else 'single_layer'
            fig.savefig(output / f'{metric}_{suffix}.png', dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        subset = frame[frame.study == 'local_window']
        for strategy, values in subset.groupby('strategy', sort=False):
            ax.plot(values.window, values[metric], marker='o', label=strategy)
        ax.axhline(baseline[metric], color='black', linestyle='--', label='baseline')
        ax.set(xlabel='Causal window size (including current token)', ylabel=label,
               title='Local attention: early/middle/late/all selected layers')
        ax.set_xscale('log', base=2)
        windows = sorted(subset.window.unique())
        ax.set_xticks(windows, labels=windows)
        if metric == 'perplexity':
            ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}_local_window.png', dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        uniform = frame[frame.study == 'uniform_region']
        regions = uniform.strategy.tolist()
        local = frame[(frame.study == 'local_window') &
                      (frame.window == min(local_window, examples['learned'].shape[-1]))]
        positions = np.arange(len(regions))
        ax.bar(positions - .18, uniform[metric], width=.36, label='uniform')
        ax.bar(positions + .18, [local[local.strategy == region].iloc[0][metric]
                                for region in regions], width=.36,
               label=f'local (window={local_window})')
        ax.axhline(baseline[metric], color='black', linestyle='--', label='baseline')
        ax.set_xticks(positions, labels=regions)
        ax.set(xlabel='Selected layer region (all heads affected)', ylabel=label,
               title='Uniform versus local attention by layer region')
        if metric == 'perplexity':
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}_region_comparison.png', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * .35)))
    heatmap = ax.imshow(scores[layers], aspect='auto', cmap='viridis')
    ax.set(xlabel='Head index', ylabel='Layer index', title='Validation head-removal importance')
    ax.set_yticks(range(len(layers)), labels=layers)
    fig.colorbar(heatmap, ax=ax, label='Mean absolute head-gate loss gradient')
    fig.tight_layout()
    fig.savefig(output / 'head_importance.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    maximum = max(values.max() for values in examples.values())
    for ax, (name, values) in zip(axes, examples.items()):
        heatmap = ax.imshow(values, vmin=0, vmax=maximum, cmap='viridis', aspect='equal', origin='upper')
        title = f'local (window={local_window})' if name == 'local' else name.replace('_', ' ')
        ax.set(xlabel='Key position', ylabel='Query position', title=title)
    fig.suptitle(f'Validation attention examples: layer {example_layer}, head {example_head}; future connections masked')
    fig.subplots_adjust(left=.05, right=.88, bottom=.16, top=.8, wspace=.35)
    colorbar_axis = fig.add_axes([.92, .2, .012, .55])
    fig.colorbar(heatmap, cax=colorbar_axis, label='Attention probability')
    fig.savefig(output / 'attention_patterns.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/wikitext2/wikitext-2-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ab5_plots')
    parser.add_argument('--layers', type=int, nargs='+', help='Zero-based layer indices')
    parser.add_argument('--eval-blocks', type=int, default=16)
    parser.add_argument('--calibration-blocks', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local-window', type=int, default=8,
                        help='Current token plus at most window-1 previous tokens')
    args = parser.parse_args()
    if min(args.eval_blocks, args.calibration_blocks, args.batch_size, args.trials,
           args.group_size, args.local_window) < 1 or args.block_size < 2:
        parser.error('Counts/windows must be positive and block-size must be at least 2.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_model()
    backbone = model[0].auto_model
    if backbone.config.model_type != 'gpt2' or not hasattr(backbone, 'h'):
        raise ValueError('This experiment supports the configured GPT-2 backbone only.')
    if getattr(backbone.config, 'add_cross_attention', False):
        raise ValueError('Cross-attention is not supported.')
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
    calibration = token_blocks(model.tokenizer, args.data_dir / 'validation-00000-of-00001.parquet',
                               args.block_size, args.calibration_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_backend = backbone.config._attn_implementation
    original_control = evaluate(backbone, test[:1], args.batch_size)
    patterns = None
    gates = None
    rows = []
    cache = {}
    try:
        backbone.set_attn_implementation('eager')
        eager_control = evaluate(backbone, test[:1], args.batch_size)
        if abs(original_control['loss'] - eager_control['loss']) > 1e-4:
            raise RuntimeError('Eager attention does not match the original backend baseline.')
        baseline = evaluate(backbone, test, args.batch_size)
        patterns = AttentionPatterns(backbone)
        patterns.set('local', candidates, args.block_size)
        full_window_control = evaluate(backbone, test[:1], args.batch_size)
        patterns.set()
        if abs(full_window_control['loss'] - eager_control['loss']) > 1e-6 or full_window_control['accuracy'] != eager_control['accuracy']:
            raise RuntimeError('Full-context local attention does not match the eager baseline.')
        gates = HeadGates(backbone)
        scores = importance_scores(backbone, gates, calibration, args.batch_size)
        gates.close()
        gates = None
        print(f'Baseline: {baseline}; backend/full-window controls passed.', flush=True)

        def run(study, mode, strategy, heads, window=args.local_window, trial=0,
                requested_percent=0., layer_index=-1):
            key = (mode, tuple(sorted(heads)), window if mode == 'local' else 0)
            if not heads or (mode == 'local' and window >= args.block_size):
                metrics = baseline
            elif key in cache:
                metrics = cache[key]
            else:
                patterns.set(mode, heads, window)
                try:
                    metrics = evaluate(backbone, test, args.batch_size)
                finally:
                    patterns.set()
                cache[key] = metrics
            rows.append(dict(study=study, mode=mode, strategy=strategy, trial=trial,
                             requested_percent=requested_percent,
                             actual_percent=100 * len(heads) / len(candidates),
                             affected_heads=len(heads), heads=json.dumps(heads),
                             window=window if mode == 'local' else 0,
                             layer_index=layer_index, delta_loss=metrics['loss'] - baseline['loss'], **metrics))
            print(f'{study} {mode}/{strategy} trial={trial + 1} heads={len(heads)} '
                  f'window={window if mode == "local" else 0} '
                  f"accuracy={metrics['accuracy']:.4f}% perplexity={metrics['perplexity']:.3f}", flush=True)

        run('baseline', 'none', 'baseline', [])
        orders = [('least_important', 0, sorted(candidates, key=lambda item: (scores[item], item))),
                  ('most_important', 0, sorted(candidates, key=lambda item: (-scores[item], item)))]
        for trial in range(args.trials):
            orders.append(('random_individual', trial, random_order(candidates, np.random.default_rng(args.seed + trial))))
            orders.append(('random_groups', trial, random_order(candidates, np.random.default_rng(args.seed + trial), args.group_size)))
        for strategy, trial, order in orders:
            for percent in PERCENTAGES:
                count = max(1, round(len(candidates) * percent / 100))
                for mode in ('uniform', 'local'):
                    run('head_budget', mode, strategy, order[:count], trial=trial, requested_percent=percent)
        for layer in layers:
            selected = [head for head in candidates if head[0] == layer]
            for mode in ('uniform', 'local'):
                run('single_layer', mode, 'single_layer', selected, layer_index=layer)
        regions = {name: group.tolist() for name, group in zip(('early', 'middle', 'late'), np.array_split(layers, 3)) if len(group)}
        regions['all'] = layers
        windows = sorted({min(args.block_size, window) for window in (1, 2, 4, 8, 16, 32, args.block_size, args.local_window)})
        for region, region_layers in regions.items():
            selected = [head for head in candidates if head[0] in region_layers]
            run('uniform_region', 'uniform', region, selected)
            for window in windows:
                run('local_window', 'local', region, selected, window=window)
        example_layer, example_head = layers[len(layers) // 2], 0
        examples = capture_examples(backbone, patterns, calibration, example_layer, example_head, args.local_window)
    finally:
        if gates is not None:
            gates.close()
        if patterns is not None:
            patterns.close()
        backbone.set_attn_implementation(original_backend)
    with (args.output_dir / 'ablation_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.output_dir / 'head_importance.npy', scores)
    np.savez(args.output_dir / 'attention_examples.npz', **examples)
    example_info = {'split': 'validation', 'layer': example_layer, 'head': example_head,
                    'input_ids': calibration[0].tolist(),
                    'tokens': model.tokenizer.convert_ids_to_tokens(calibration[0].tolist())}
    (args.output_dir / 'attention_example_tokens.json').write_text(json.dumps(example_info, indent=2))
    metadata = {**vars(args), 'data_dir': str(args.data_dir), 'output_dir': str(args.output_dir),
                'layers': layers, 'model_type': backbone.config.model_type,
                'original_attention_backend': original_backend, 'experiment_attention_backend': 'eager',
                'backend_baseline_verified': True, 'full_window_baseline_verified': True,
                'test_tokens': baseline['tokens'], 'calibration_tokens': calibration.shape[0] * (args.block_size - 1),
                'percentage_budgets': PERCENTAGES, 'window_sweep': windows,
                'uniform': 'equal probabilities over causally available keys; no future tokens',
                'local': 'learned probabilities renormalized over current token and window-1 preceding tokens',
                'importance': 'token-weighted mean absolute validation loss gradient w.r.t. head-output gates, as in Experiment A',
                'group_policy': 'random head groups within layers; partial final group to match exact head budget',
                'region_policy': 'selected layers divided into three ordered near-equal groups',
                'paired_policy': 'uniform and local interventions use identical head subsets; random windows use no extra randomness',
                'plot_policy': 'one observation per trial and actual budget; error bars are sample standard deviations',
                'limitations': 'Overall next-token prediction only. Head-removal importance need not predict importance for attention-pattern replacement. No caching/cross-attention/training/upcast mode.'}
    (args.output_dir / 'experiment_config.json').write_text(json.dumps(metadata, indent=2))
    save_plots(rows, scores, layers, examples, example_layer, example_head, args.local_window, args.output_dir)
    print(f'{len(rows)} results and all plots saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
