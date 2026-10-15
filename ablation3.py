"""Experiment C: GPT-2 input-token embedding ablation on held-out WikiText-2.

Run: venv/bin/python ablation3.py
Experiment A defaults: 16 test blocks, 8 validation blocks, 64 tokens/block,
batch size 2, 3 trials, group size 3, seed 42; budgets 5/10/20/30/50/75%.
All results and plots default to ab3_plots/.

Only token embeddings are changed via inputs_embeds. GPT-2 adds its original
positional embeddings afterward. Labels, tied output weights, and sequence
lengths stay intact. EOS and the last (unscored) input position are excluded.
Content/function/marker categories are transparent lexical heuristics, not POS
tags. Metrics measure overall language modeling, not a dedicated semantic task.
Percentages are fractions of eligible tokens in the target category; paired
random controls match the actual token count separately in every context.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ablation1 import PERCENTAGES, evaluate, token_blocks
from model import load_model

ROOT = Path(__file__).resolve().parent
FUNCTION_WORDS = frozenset('''a an the this that these those i me my mine myself we us
our ours ourselves you your yours yourself yourselves he him his himself she her
hers herself it its itself they them their theirs themselves who whom whose which
what whoever whatever am is are was were be being been do does did have has had
can could may might must shall should will would ought to of in on at by for with
from as into onto upon over under above below between among through during before
after since until about against without within around than and or but nor yet so
if because although though while whereas unless whether both either neither each
every all any some no not only also even just very too such there here when where
why how once twice more most less least much many few several another other
same own then now up down out off again ever never always often sometimes
don't doesn't didn't isn't aren't wasn't weren't can't couldn't won't wouldn't
shouldn't haven't hasn't hadn't mustn't needn't i'm you're he's she's it's we're
they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll
he'll she'll we'll they'll that's there's here's what's who's let's
'''.split())
MARKER_RE = re.compile(r"[.,;:!?()\[\]{}\"'’\-–—]+")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
CATEGORIES = ('content', 'function', 'markers')
MODES = ('zero', 'mean', 'noise')


def categorize_tokens(tokenizer, blocks):
    """Reconstruct words across BPE fragments before assigning token categories.

    A word cut by a context boundary can be incomplete. Alphabetic words absent
    from FUNCTION_WORDS are content candidates; punctuation is a marker proxy.
    Unclassified numbers/whitespace remain eligible for random controls.
    """
    eligible = torch.ones_like(blocks, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        eligible &= blocks != token_id
    eligible[:, -1] = False
    masks = {name: torch.zeros_like(eligible) for name in CATEGORIES}
    for row, ids in enumerate(blocks.tolist()):
        pieces = [tokenizer.decode([token], clean_up_tokenization_spaces=False,
                                   skip_special_tokens=False) for token in ids]
        ends = np.cumsum([len(piece) for piece in pieces])
        starts = np.concatenate(([0], ends[:-1]))
        text = ''.join(pieces)
        for match in WORD_RE.finditer(text):
            word = match.group().lower().replace('’', "'")
            category = 'function' if word in FUNCTION_WORDS else 'content'
            for col in np.flatnonzero((starts < match.end()) & (ends > match.start())):
                if eligible[row, col]:
                    masks[category][row, col] = True
        for match in MARKER_RE.finditer(text):
            for col in np.flatnonzero((starts < match.end()) & (ends > match.start())):
                # Word tokens take precedence, including contractions.
                if eligible[row, col] and not masks['content'][row, col] and not masks['function'][row, col]:
                    masks['markers'][row, col] = True
    return eligible, masks


@torch.no_grad()
def reference_statistics(backbone, calibration):
    device = next(backbone.parameters()).device
    embeddings = backbone.get_input_embeddings()(calibration.to(device)).float()
    reference = embeddings.mean(dim=(0, 1))
    rms = (embeddings - reference).square().mean().sqrt().item()
    if rms <= 0:
        raise ValueError('Validation embeddings have zero variance; noise cannot be scaled.')
    return reference, rms


@torch.no_grad()
def evaluate_embeddings(backbone, blocks, embeddings, mask, mode, reference,
                        noise, strength, batch_size):
    device = next(backbone.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    for start in range(0, len(blocks), batch_size):
        end = start + batch_size
        values = embeddings[start:end].to(device).clone()
        selected = mask[start:end].to(device)
        if mode == 'zero':
            values[selected] = 0
        elif mode == 'mean':
            values[selected] = reference.to(device=device, dtype=values.dtype)
        elif mode == 'noise':
            values += selected.unsqueeze(-1) * noise[start:end].to(device=device, dtype=values.dtype) * strength
        else:
            raise ValueError(f'Unknown mode: {mode}')
        hidden = backbone(inputs_embeds=values, use_cache=False).last_hidden_state
        logits = F.linear(hidden[:, :-1], backbone.get_input_embeddings().weight)
        labels = blocks[start:end, 1:].to(device)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    labels.reshape(-1), reduction='sum').item()
    loss = loss_sum / total
    return dict(accuracy=100 * correct / total, loss=loss,
                perplexity=math.exp(min(loss, 700)), tokens=total)


def ranked_positions(mask, rng, group_size=1):
    orders = []
    for row in mask.numpy():
        positions = np.flatnonzero(row)
        if group_size == 1:
            orders.append(rng.permutation(positions))
        else:
            # Consecutive eligible positions in chunks; exact budgets can split
            # the final chunk, matching Experiment A's partial-group policy.
            groups = [positions[i:i + group_size] for i in range(0, len(positions), group_size)]
            orders.append(np.concatenate([groups[i] for i in rng.permutation(len(groups))])
                          if groups else positions)
    return orders


def mask_from_counts(shape, orders, counts):
    mask = torch.zeros(shape, dtype=torch.bool)
    for row, (order, count) in enumerate(zip(orders, counts)):
        mask[row, order[:int(count)]] = True
    return mask


def masks_for_trial(eligible, categories, trial, seed, group_size):
    rng = np.random.default_rng(seed + trial)
    random_order = ranked_positions(eligible, rng)
    for category, candidates in categories.items():
        targeted_order = ranked_positions(candidates, rng)
        sizes = candidates.sum(1).numpy()
        for percent in PERCENTAGES:
            counts = np.where(sizes > 0, np.maximum(1, np.rint(sizes * percent / 100)), 0).astype(int)
            targeted = mask_from_counts(eligible.shape, targeted_order, counts)
            random_mask = mask_from_counts(eligible.shape, random_order, counts)
            yield category, 'targeted', percent, targeted
            yield category, 'random_matched', percent, random_mask
    grouped_order = ranked_positions(eligible, rng, group_size)
    sizes = eligible.sum(1).numpy()
    for percent in PERCENTAGES:
        counts = np.where(sizes > 0, np.maximum(1, np.rint(sizes * percent / 100)), 0).astype(int)
        yield 'all', 'random_individual', percent, mask_from_counts(eligible.shape, random_order, counts)
        yield 'all', 'random_groups', percent, mask_from_counts(eligible.shape, grouped_order, counts)


def save_plots(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    baseline = frame[frame.strategy == 'baseline'].iloc[0]
    for metric, label in [('accuracy', 'Next-token accuracy (%)'), ('perplexity', 'Perplexity')]:
        for category in (*CATEGORIES, 'all'):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, mode in zip(axes, MODES):
                subset = frame[(frame.category == category) & (frame['mode'] == mode) & (frame.study == 'mask_budget')]
                for strategy, values in subset.groupby('strategy', sort=False):
                    summary = values.groupby('requested_percent')[metric].agg(['mean', 'std'])
                    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'].fillna(0),
                                marker='o', capsize=3, label=strategy.replace('_', ' '))
                ax.axhline(baseline[metric], linestyle='--', color='black', label='baseline')
                ax.set(xlabel=f'{category} eligible token budget (%)', ylabel=label,
                       title=mode if mode != 'noise' else 'noise (0.5 × validation RMS)')
                if metric == 'perplexity':
                    ax.set_yscale('log')
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            fig.suptitle(f'GPT-2 input-embedding ablation: {category}')
            fig.tight_layout()
            fig.savefig(output / f'{metric}_{category}_ablation.png', dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        values = frame[frame.study == 'noise_strength']
        summary = values.groupby('noise_scale')[metric].agg(['mean', 'std'])
        ax.errorbar(summary.index, summary['mean'], yerr=summary['std'].fillna(0), marker='o', capsize=3)
        ax.axhline(baseline[metric], linestyle='--', color='black', label='baseline')
        ax.set(xlabel='Gaussian noise standard deviation / validation embedding RMS',
               ylabel=label, title='Noise strength: random masking of 50% of eligible input tokens')
        if metric == 'perplexity':
            ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}_noise_strength.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset/wikitext2/wikitext-2-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ab3_plots')
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
    if backbone.config.model_type != 'gpt2':
        raise ValueError('This experiment supports the configured GPT-2 backbone only.')
    if args.block_size > backbone.config.n_positions:
        parser.error('block-size exceeds the model context length.')
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.config.output_attentions = False
    backbone.config.output_hidden_states = False
    test = token_blocks(model.tokenizer, args.data_dir / 'test-00000-of-00001.parquet',
                        args.block_size, args.eval_blocks)
    calibration = token_blocks(model.tokenizer, args.data_dir / 'validation-00000-of-00001.parquet',
                               args.block_size, args.calibration_blocks)
    eligible, categories = categorize_tokens(model.tokenizer, test)
    reference, rms = reference_statistics(backbone, calibration)
    device = next(backbone.parameters()).device
    with torch.no_grad():
        embeddings = backbone.get_input_embeddings()(test.to(device)).cpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(backbone, test, args.batch_size)
    empty = torch.zeros_like(eligible)
    zeros = torch.zeros_like(embeddings)
    intact = evaluate_embeddings(backbone, test, embeddings, empty, 'zero', reference, zeros, 0, args.batch_size)
    if abs(intact['loss'] - baseline['loss']) > 1e-5 or intact['accuracy'] != baseline['accuracy']:
        raise RuntimeError('inputs_embeds baseline does not match the input_ids baseline.')
    rows = []
    denominator = int(eligible.sum())
    if denominator == 0:
        raise ValueError('No eligible input tokens to ablate.')

    def record(study, mode, category, strategy, trial, percent, mask, noise_scale, metrics):
        rows.append(dict(study=study, mode=mode, category=category, strategy=strategy,
                         trial=trial, requested_percent=percent,
                         actual_percent=100 * int(mask.sum()) / denominator,
                         masked_tokens=int(mask.sum()), eligible_tokens=denominator,
                         category_tokens=int(categories[category].sum()) if category in categories else denominator,
                         noise_scale=noise_scale, **metrics))

    record('baseline', 'none', 'all', 'baseline', 0, 0, empty, 0, baseline)
    np.save(args.output_dir / 'reference_embedding.npy', reference.cpu().numpy())
    print(f'Baseline: {baseline}; validation RMS={rms:.6f}; '
          f'category counts={ {key: int(value.sum()) for key,value in categories.items()} }', flush=True)
    cache = {}
    for trial in range(args.trials):
        generator = torch.Generator(device='cpu').manual_seed(args.seed + 10000 + trial)
        noise = torch.randn(embeddings.shape, generator=generator) * rms
        for category, strategy, percent, mask in masks_for_trial(eligible, categories, trial, args.seed, args.group_size):
            if not mask.any():
                continue
            for mode in MODES:
                scale = .5 if mode == 'noise' else 0
                key = (mode, mask.numpy().tobytes(), trial if mode == 'noise' else -1, scale)
                if key not in cache:
                    cache[key] = evaluate_embeddings(backbone, test, embeddings, mask, mode,
                                                      reference, noise, scale, args.batch_size)
                metrics = cache[key]
                record('mask_budget', mode, category, strategy, trial, percent, mask, scale, metrics)
                print(f'{mode} {category}/{strategy} trial={trial + 1} budget={percent}% '
                      f"masked={int(mask.sum())} accuracy={metrics['accuracy']:.4f}% "
                      f"perplexity={metrics['perplexity']:.3f}", flush=True)
        rng = np.random.default_rng(args.seed + trial)
        orders = ranked_positions(eligible, rng)
        sizes = eligible.sum(1).numpy()
        counts = np.where(sizes > 0, np.maximum(1, np.rint(sizes * .5)), 0).astype(int)
        mask = mask_from_counts(test.shape, orders, counts)
        for scale in (0, .1, .5, 1.0):
            key = ('noise', mask.numpy().tobytes(), trial, scale)
            if scale == 0:
                metrics = baseline
            else:
                if key not in cache:
                    cache[key] = evaluate_embeddings(backbone, test, embeddings, mask, 'noise',
                                                      reference, noise, scale, args.batch_size)
                metrics = cache[key]
            record('noise_strength', 'noise', 'all', 'random_individual', trial, 50, mask, scale, metrics)
    with (args.output_dir / 'ablation_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {**vars(args), 'data_dir': str(args.data_dir), 'output_dir': str(args.output_dir),
                'model_type': backbone.config.model_type, 'percentage_budgets': PERCENTAGES,
                'test_tokens': baseline['tokens'], 'calibration_input_tokens': calibration.numel(),
                'reference': 'token-frequency-weighted validation input embedding mean',
                'noise': 'independent Gaussian noise; standard deviation = noise_scale * validation centered embedding RMS',
                'validation_embedding_rms': rms, 'noise_scales': [0, .1, .5, 1.0],
                'positional_embeddings_preserved': True, 'labels_and_output_weights_preserved': True,
                'inputs_embeds_baseline_verified': True,
                'category_counts': {key: int(value.sum()) for key, value in categories.items()},
                'function_word_lexicon': sorted(FUNCTION_WORDS),
                'category_policy': 'reconstructed alphabetic words: explicit function lexicon versus other content candidates; punctuation-only marker proxy; categories are heuristic',
                'mask_policy': 'nested token-occurrence subsets per trial; targeted and random masks have identical counts per block; exclude special tokens and final input position',
                'group_policy': 'chunks of group_size consecutive eligible positions; partial final group to match exact budget',
                'limitations': 'No POS parser or semantic-task evaluation. Word fragments at context boundaries and Unicode byte fragments can affect heuristic categories.'}
    (args.output_dir / 'experiment_config.json').write_text(json.dumps(metadata, indent=2))
    save_plots(rows, args.output_dir)
    print(f'{len(rows)} results and all plots saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
