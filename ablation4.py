"""Experiment D: GPT-2 residual-stream and residual-update ablation.

Run: venv/bin/python ablation4.py
Selected layers: venv/bin/python ablation4.py --layers 0 1 2
Default outputs: ab4_plots/. Model/data/evaluation defaults match ablation1.py.

block: h_out = h_in + alpha * (original_block_output - h_in).
attention/MLP: scale their outputs immediately before their residual addition.
Alpha=0 removes that contribution; alpha=1 preserves it. Attention interventions
also change the input to the subsequent MLP. Scaling a complete block's net
update is therefore distinct from scaling both sublayer updates independently.

Dimensions: suppress selected update coordinates (block/attention/MLP), or zero
selected coordinates of the full outgoing residual stream (residual_stream).
The latter also removes carried-forward information. Shared coordinate masks
across selected layers allow equal-budget comparisons. Input embeddings,
positions, learned weights, and the final model normalization remain intact.

Studies: individual-layer contribution removal; alpha sweeps for early/middle/
late/all selected layers; increasing cumulative removal of validation-ranked or
random updates; validation-ranked/random residual-dimension ablation.

Research question: accumulated small updates versus disproportionately important
updates. Rankings use held-out validation loss gradients, never the test set.
Evaluation matches Experiment A: overall next-token accuracy/loss/perplexity.
The referenced four task categories are unspecified; this script does not claim
to evaluate additional semantic, syntactic, or other task-specific benchmarks.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ablation1 import PERCENTAGES, evaluate, logits_for, token_blocks
from model import load_model

ROOT = Path(__file__).resolve().parent
COMPONENTS = ('block', 'attention', 'mlp')
DIMENSION_COMPONENTS = (*COMPONENTS, 'residual_stream')
ALPHAS = (0.0, .25, .5, .75, 1.0)


def first_tensor(output):
    return output if torch.is_tensor(output) else output[0]


def replace_tensor(output, value):
    return value if torch.is_tensor(output) else (value, *output[1:])


class ResidualGates:
    """Temporary, differentiable gates; model weights are never modified."""
    def __init__(self, backbone):
        self.layers = len(backbone.h)
        self.width = backbone.config.n_embd
        device = next(backbone.parameters()).device
        self.gates = torch.ones(self.layers, self.width, device=device, requires_grad=True)
        self.component = None
        self.inputs = {}
        self.handles = []
        try:
            for layer, block in enumerate(backbone.h):
                def capture(module, args, kwargs, layer=layer):
                    if self.component == 'block':
                        self.inputs[layer] = args[0] if args else kwargs['hidden_states']

                def block_output(module, args, output, layer=layer):
                    value = first_tensor(output)
                    if self.component == 'block':
                        incoming = self.inputs.pop(layer)
                        gate = self.gates[layer].to(dtype=value.dtype)
                        # Preserve alpha=1 exactly. alpha=0 returns incoming
                        # exactly, avoiding subtraction roundoff at the endpoint.
                        scaled = value + (gate - 1) * (value - incoming)
                        scaled = torch.where(gate == 0, incoming, scaled)
                        return replace_tensor(output, scaled)
                    if self.component == 'residual_stream':
                        return replace_tensor(output, value * self.gates[layer].to(dtype=value.dtype))
                    return output

                def sublayer_output(kind, layer):
                    def hook(module, args, output):
                        if self.component != kind:
                            return output
                        value = first_tensor(output)
                        return replace_tensor(output, value * self.gates[layer].to(dtype=value.dtype))
                    return hook

                self.handles.append(block.register_forward_pre_hook(capture, with_kwargs=True))
                self.handles.append(block.register_forward_hook(block_output))
                self.handles.append(block.attn.register_forward_hook(sublayer_output('attention', layer)))
                self.handles.append(block.mlp.register_forward_hook(sublayer_output('mlp', layer)))
        except Exception:
            self.close()
            raise

    def set(self, component=None, layers=(), alpha=1.0, dimensions=None):
        if component is not None and component not in DIMENSION_COMPONENTS:
            raise ValueError(f'Unknown component: {component}')
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1.')
        if any(layer < 0 or layer >= self.layers for layer in layers):
            raise ValueError('Invalid layer index.')
        if dimensions is not None and any(dim < 0 or dim >= self.width for dim in dimensions):
            raise ValueError('Invalid dimension index.')
        self.component = component
        self.inputs.clear()
        with torch.no_grad():
            self.gates.fill_(1)
            for layer in layers:
                if dimensions is None:
                    self.gates[layer].fill_(alpha)
                else:
                    self.gates[layer, dimensions] = alpha

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.inputs.clear()
        self.component = None


def importance_scores(backbone, gates, blocks, batch_size):
    """Token-weighted abs gradients at unaltered gate=1.

    Layer importance: abs(sum of coordinate gradients), the derivative for a
    scalar layer gate. Dimension importance: mean absolute coordinate gradient
    over selected layers (the averaging is done after calibration).
    """
    device = next(backbone.parameters()).device
    layer_scores = {}
    dimension_scores = {}
    for component in DIMENSION_COMPONENTS:
        gates.set(component)
        per_layer = torch.zeros(gates.layers, device=device)
        per_dimension = torch.zeros_like(gates.gates)
        total = 0
        for batch in blocks.split(batch_size):
            gates.gates.grad = None
            batch = batch.to(device)
            logits = logits_for(backbone, batch)
            labels = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            gradient = gates.gates.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f'Invalid importance gradients for {component}.')
            tokens = labels.numel()
            per_layer += gradient.detach().sum(-1).abs() * tokens
            per_dimension += gradient.detach().abs() * tokens
            total += tokens
        layer_scores[component] = (per_layer / total).cpu().numpy()
        dimension_scores[component] = (per_dimension / total).cpu().numpy()
        print(f'Validation importance calibrated: {component}', flush=True)
    gates.gates.grad = None
    gates.gates.requires_grad_(False)
    gates.set()
    return layer_scores, dimension_scores


def layer_orders(layers, scores, trials, seed, group_size):
    yield 'least_important', 0, sorted(layers, key=lambda layer: (scores[layer], layer))
    yield 'most_important', 0, sorted(layers, key=lambda layer: (-scores[layer], layer))
    groups = [layers[i:i + group_size] for i in range(0, len(layers), group_size)]
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        yield 'random_individual', trial, [int(layer) for layer in rng.permutation(layers)]
        yield 'random_groups', trial, [layer for index in rng.permutation(len(groups)) for layer in groups[index]]


def dimension_orders(scores, trials, seed):
    yield 'least_important', 0, np.argsort(scores, kind='stable').tolist()
    yield 'most_important', 0, np.argsort(-scores, kind='stable').tolist()
    for trial in range(trials):
        yield 'random', trial, np.random.default_rng(seed + trial).permutation(len(scores)).tolist()


def save_plots(rows, layer_scores, dimension_scores, layers, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    baseline = frame[frame.study == 'baseline'].iloc[0]
    studies = [('single_layer', 'layer_index', 'Removed update at layer index'),
               ('alpha_scaling', 'alpha', 'Residual update scale alpha'),
               ('cumulative_removal', 'actual_percent', 'Removed updates in selected layers (%)'),
               ('dimension_ablation', 'actual_percent', 'Ablated residual dimensions (%)')]
    for metric, label in [('accuracy', 'Next-token accuracy (%)'), ('perplexity', 'Perplexity')]:
        for study, x, xlabel in studies:
            components = DIMENSION_COMPONENTS if study == 'dimension_ablation' else COMPONENTS
            fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 4.7))
            for ax, component in zip(axes, components):
                subset = frame[(frame.study == study) & (frame.component == component)]
                for strategy, values in subset.groupby('strategy', sort=False):
                    # Rounded percentages can share the same actual budget.
                    # Count each trial once so repeated CSV budgets do not
                    # artificially shrink its plotted standard deviation.
                    values = values.drop_duplicates(subset=['trial', x])
                    summary = values.groupby(x)[metric].agg(['mean', 'std'])
                    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'].fillna(0),
                                marker='o', capsize=3, label=strategy.replace('_', ' '))
                ax.axhline(baseline[metric], linestyle='--', color='black', label='baseline')
                ax.set(xlabel=xlabel, ylabel=label, title=component.replace('_', ' '))
                if metric == 'perplexity':
                    ax.set_yscale('log')
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            fig.suptitle(f'GPT-2 residual ablation: {study.replace("_", " ")}')
            fig.tight_layout()
            fig.savefig(output / f'{metric}_{study}.png', dpi=180)
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, component in zip(axes, COMPONENTS):
        ax.bar(layers, layer_scores[component][layers])
        ax.set(xlabel='Layer index', ylabel='Mean absolute scalar-gate loss gradient', title=component)
        ax.set_xticks(layers)
    fig.suptitle('Validation residual-update importance')
    fig.tight_layout()
    fig.savefig(output / 'layer_importance.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(13, 8))
    for ax, component in zip(axes, DIMENSION_COMPONENTS):
        scores = dimension_scores[component][layers].mean(0)
        ax.plot(range(len(scores)), scores, linewidth=.8)
        ax.set(xlabel='Residual dimension index', ylabel='Mean |loss gradient|', title=component.replace('_', ' '))
    fig.tight_layout()
    fig.savefig(output / 'dimension_importance.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/wikitext2/wikitext-2-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ab4_plots')
    parser.add_argument('--layers', type=int, nargs='+', help='Zero-based layer indices')
    parser.add_argument('--eval-blocks', type=int, default=16)
    parser.add_argument('--calibration-blocks', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=3)
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
    if getattr(backbone.config, 'add_cross_attention', False):
        raise ValueError('Cross-attention GPT-2 blocks are not supported by this study.')
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
    calibration = token_blocks(model.tokenizer, args.data_dir / 'validation-00000-of-00001.parquet',
                               args.block_size, args.calibration_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(backbone, test, args.batch_size)
    gates = ResidualGates(backbone)
    rows = []
    cache = {}

    def run(study, component, strategy, selected, alpha=0., dimensions=None,
            trial=0, requested_percent=0.):
        key = (component, tuple(sorted(selected)), alpha,
               None if dimensions is None else tuple(sorted(dimensions)))
        if alpha == 1 or not selected or dimensions == []:
            metrics = baseline
        elif key in cache:
            metrics = cache[key]
        else:
            gates.set(component, selected, alpha, dimensions)
            try:
                metrics = evaluate(backbone, test, args.batch_size)
            finally:
                gates.set()
            cache[key] = metrics
        count = len(selected) if dimensions is None else len(dimensions)
        denominator = len(layers) if dimensions is None else gates.width
        rows.append(dict(study=study, component=component, strategy=strategy, trial=trial,
                         alpha=alpha, requested_percent=requested_percent,
                         actual_percent=100 * count / denominator, affected_layers=len(selected),
                         ablated_dimensions=0 if dimensions is None else len(dimensions),
                         layers=json.dumps(selected), dimensions=json.dumps(dimensions or []),
                         layer_index=selected[0] if study == 'single_layer' else -1,
                         delta_loss=metrics['loss'] - baseline['loss'], **metrics))
        print(f'{study} {component}/{strategy} alpha={alpha:g} layers={len(selected)} '
              f'dims={0 if dimensions is None else len(dimensions)} trial={trial + 1} '
              f"accuracy={metrics['accuracy']:.4f}% perplexity={metrics['perplexity']:.3f}", flush=True)

    try:
        run('baseline', 'none', 'baseline', [], alpha=1.)
        # Verify all-one gates reproduce the unhooked model, including the full
        # block interpolation endpoint, before calibration and baseline reuse.
        for component in DIMENSION_COMPONENTS:
            gates.set(component, layers, alpha=1.)
            control = evaluate(backbone, test[:1], args.batch_size)
            gates.set()
            reference = evaluate(backbone, test[:1], args.batch_size)
            if abs(control['loss'] - reference['loss']) > 1e-6 or control['accuracy'] != reference['accuracy']:
                raise RuntimeError(f'alpha=1 baseline check failed for {component}.')
        layer_scores, dimension_scores = importance_scores(backbone, gates, calibration, args.batch_size)
        for component in COMPONENTS:
            for layer in layers:
                run('single_layer', component, 'single_layer', [layer])
            regions = {name: group.tolist() for name, group in zip(('early', 'middle', 'late'), np.array_split(layers, 3)) if len(group)}
            regions['all'] = layers
            for region, selected in regions.items():
                for alpha in ALPHAS:
                    run('alpha_scaling', component, region, selected, alpha=alpha)
            for strategy, trial, order in layer_orders(layers, layer_scores[component], args.trials,
                                                        args.seed, args.group_size):
                for percent in PERCENTAGES:
                    count = max(1, round(len(layers) * percent / 100))
                    run('cumulative_removal', component, strategy, order[:count],
                        trial=trial, requested_percent=percent)
        for component in DIMENSION_COMPONENTS:
            scores = dimension_scores[component][layers].mean(0)
            for strategy, trial, order in dimension_orders(scores, args.trials, args.seed):
                for percent in PERCENTAGES:
                    count = max(1, round(gates.width * percent / 100))
                    run('dimension_ablation', component, strategy, layers,
                        dimensions=order[:count], trial=trial, requested_percent=percent)
    finally:
        gates.close()
    with (args.output_dir / 'ablation_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(args.output_dir / 'residual_importance.npz',
             **{f'layer_{name}': values for name, values in layer_scores.items()},
             **{f'dimension_{name}': values for name, values in dimension_scores.items()})
    metadata = {**vars(args), 'data_dir': str(args.data_dir), 'output_dir': str(args.output_dir),
                'layers': layers, 'model_type': backbone.config.model_type,
                'test_tokens': baseline['tokens'], 'calibration_tokens': calibration.shape[0] * (args.block_size - 1),
                'percentage_budgets': PERCENTAGES, 'alphas': ALPHAS,
                'alpha_one_baselines_verified': True,
                'block_equation': 'h_out = h_in + alpha * (original_block_output - h_in)',
                'attention_intervention': 'scale attention output after projection/dropout and before residual addition; downstream MLP sees modified state',
                'mlp_intervention': 'scale MLP output after projection/dropout and before residual addition',
                'dimension_policy': 'same coordinate mask in selected layers; suppress update coordinates for block/attention/MLP; zero full outgoing coordinates for residual_stream',
                'layer_importance': 'token-weighted mean abs(sum of coordinate gate gradients) on validation data at alpha=1',
                'dimension_importance': 'token-weighted mean absolute coordinate gate gradient, averaged across selected layers for ranking',
                'group_policy': 'chunks of group_size selected layers; randomized group order, partial final group to match layer budget',
                'region_policy': 'ordered selected layers split into three near-equal early/middle/late groups',
                'plot_policy': 'one observation per trial and actual budget; error bars show sample standard deviation',
                'task': 'WikiText-2 next-token prediction',
                'limitations': 'The four referenced task categories were not specified. No additional task-specific benchmark is evaluated. Gradient scores are local sensitivity estimates.'}
    (args.output_dir / 'experiment_config.json').write_text(json.dumps(metadata, indent=2))
    save_plots(rows, layer_scores, dimension_scores, layers, args.output_dir)
    print(f'{len(rows)} results and all plots saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
