"""Print raw WikiText-2 statistics, optionally saving them as JSON.

Usage:
    venv/bin/python dataset_statistics.py
    venv/bin/python dataset_statistics.py --output dataset_statistics.json

Words are whitespace-separated tokens; vocabulary is case-sensitive and includes
punctuation. Counts are computed before cleaning and are not GPT-2 token counts.
Each parquet row is a text entry, not necessarily a sentence or document.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'dataset/wikitext2/wikitext-2-v1'


def describe_lengths(values):
    if not values:
        return dict(min=0, mean=0.0, median=0.0, p95=0.0, max=0)
    series = pd.Series(values)
    return dict(min=int(series.min()), mean=float(series.mean()),
                median=float(series.median()), p95=float(series.quantile(.95)),
                max=int(series.max()))


def split_statistics(paths):
    frequencies = Counter()
    text_counts = Counter()
    word_lengths, character_lengths = [], []
    rows = missing = empty = whitespace_only = characters = words = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=['text'])
        rows += len(frame)
        for text in frame['text']:
            if pd.isna(text):
                missing += 1
                continue
            if not isinstance(text, str):
                raise TypeError(f'{path}: expected text strings, got {type(text).__name__}')
            if not text:
                empty += 1
                continue
            if not text.strip():
                whitespace_only += 1
                continue
            tokens = text.split()
            frequencies.update(tokens)
            text_counts[text] += 1
            words += len(tokens)
            characters += len(text)
            word_lengths.append(len(tokens))
            character_lengths.append(len(text))
    result = dict(
        files=[str(path) for path in paths],
        file_size_bytes=sum(path.stat().st_size for path in paths),
        rows=rows, missing_rows=missing, empty_rows=empty,
        whitespace_only_rows=whitespace_only, nonempty_rows=len(word_lengths),
        total_words=words, total_characters=characters,
        unique_words=len(frequencies),
        duplicate_nonempty_rows=sum(count - 1 for count in text_counts.values()),
        words_per_nonempty_row=describe_lengths(word_lengths),
        characters_per_nonempty_row=describe_lengths(character_lengths),
        most_common_words=[dict(word=word, count=count)
                           for word, count in frequencies.most_common(10)],
    )
    return result, frequencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=Path, help='Optional JSON output path')
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f'Dataset directory does not exist: {args.data_dir}')
    results = {}
    vocabularies = {}
    for split in ('train', 'validation', 'test'):
        paths = sorted(args.data_dir.glob(f'{split}-*.parquet'))
        if not paths:
            parser.error(f'No {split} parquet files found in {args.data_dir}')
        results[split], vocabularies[split] = split_statistics(paths)
    overall_vocabulary = set().union(*(set(words) for words in vocabularies.values()))
    totals = {key: sum(stats[key] for stats in results.values())
              for key in ('rows', 'missing_rows', 'empty_rows', 'whitespace_only_rows',
                          'nonempty_rows', 'total_words', 'total_characters', 'file_size_bytes')}
    totals['unique_words'] = len(overall_vocabulary)
    coverage = {}
    train_vocabulary = set(vocabularies['train'])
    for split in ('validation', 'test'):
        unseen = set(vocabularies[split]) - train_vocabulary
        unseen_count = sum(vocabularies[split][word] for word in unseen)
        coverage[split] = dict(unique_words_absent_from_train=len(unseen),
                               word_occurrences_absent_from_train=unseen_count,
                               unseen_word_percentage=100 * unseen_count / results[split]['total_words']
                               if results[split]['total_words'] else 0.0)
    print(f'Dataset: {args.data_dir.resolve()}')
    print('Raw text; whitespace-separated, case-sensitive words (including punctuation).')
    print('Length summaries exclude missing, empty, and whitespace-only entries.\n')
    table = pd.DataFrame({split: {key: stats[key] for key in totals}
                          for split, stats in results.items()})
    table['total'] = pd.Series(totals)
    print(table.to_string())
    for split, stats in results.items():
        print(f'\n{split.upper()}')
        print(f"Duplicate nonempty rows beyond first occurrence: {stats['duplicate_nonempty_rows']}")
        for label, key in [('Words per entry', 'words_per_nonempty_row'),
                           ('Characters per entry', 'characters_per_nonempty_row')]:
            print(f"{label}: " + ', '.join(f'{name}={value:.2f}'
                                         for name, value in stats[key].items()))
        print('Top 10 words: ' + ', '.join(f"{item['word']!r}: {item['count']}"
                                         for item in stats['most_common_words']))
        if split in coverage:
            print(f"Words absent from train: {coverage[split]['unique_words_absent_from_train']} unique; "
                  f"{coverage[split]['unseen_word_percentage']:.2f}% of word occurrences")
    report = dict(data_dir=str(args.data_dir.resolve()),
                  word_definition='Whitespace-separated, case-sensitive; punctuation retained',
                  length_population='Nonempty text rows only',
                  splits=results, totals=totals, vocabulary_coverage=coverage)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\nSaved report to {args.output.resolve()}')


if __name__ == '__main__':
    main()
