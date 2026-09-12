"""
A REAL word2vec, trained on the exact same corpus (text8), same vocab
size (30k), same window (5), same epochs (8) as our wavembed checkpoints
— the fair, apples-to-apples baseline the benchmark table was missing.
Literature numbers (0.6-0.7 on WordSim-353) come from word2vec trained on
~100 BILLION words (Google News); this trains on the same ~15M words we
gave wavembed, so any gap or lead is about the representation itself
(wave vs vector), not about who saw more data.

Usage:
    python -m wavembed.word2vec_baseline --dims 6,12,24,48
"""

from __future__ import annotations

import argparse
import csv
import time

from gensim.models import Word2Vec
from scipy.stats import spearmanr

from .benchmark import DATASETS, load_pairs


def evaluate(model, pairs):
    model_scores, human_scores, missing = [], [], 0
    for w1, w2, human in pairs:
        if w1 not in model.wv or w2 not in model.wv:
            missing += 1
            continue
        model_scores.append(model.wv.similarity(w1, w2))
        human_scores.append(human)
    if len(model_scores) < 2:
        return None, 0, missing
    rho, _ = spearmanr(model_scores, human_scores)
    return rho, len(model_scores), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=str, default="6,12,24,48",
                         help="comma-separated vector sizes to train, matching wavembed's "
                              "numbers-per-word so each row is a direct apples-to-apples comparison")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-count", type=int, default=20, help="tuned to give ~vocab-size words on text8")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--negative", type=int, default=5)
    args = parser.parse_args()

    from datasets import load_dataset
    print("loading text8...")
    text = load_dataset("afmck/text8")["train"][0]["text"]
    words = text.split()
    # gensim's C backend silently truncates any single "sentence" past ~10k
    # tokens (MAX_SENTENCE_LEN) — text8 is one giant unbroken string, so
    # without chunking, most of the corpus would never actually be seen
    # during training (same reason our own corpus.py chunks text8 too).
    chunk_size = 1000
    sentences = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    dims = [int(d) for d in args.dims.split(",")]
    results = {}
    for dim in dims:
        t0 = time.time()
        model = Word2Vec(sentences, vector_size=dim, window=args.window, min_count=args.min_count,
                          sg=1, negative=args.negative, epochs=args.epochs, workers=14)
        elapsed = time.time() - t0
        print(f"\nword2vec, {dim} dims: vocab={len(model.wv)} | trained in {elapsed:.1f}s")

        row = {}
        for name, path in DATASETS.items():
            pairs = load_pairs(path)
            rho, covered, missing = evaluate(model, pairs)
            row[name] = (rho, covered, missing)
            rho_str = f"{rho:.3f}" if rho is not None else "n/a"
            print(f"  {name:<28} rho={rho_str:>7}  covered={covered:>4}  oov={missing:>4}")
        results[dim] = row

    print("\n=== summary: word2vec (real, same text8 corpus) ===")
    header = f"{'dims':<6}" + "".join(f"{name:>28}" for name in DATASETS)
    print(header)
    for dim in dims:
        line = f"{dim:<6}"
        for name in DATASETS:
            rho = results[dim][name][0]
            line += f"{(f'{rho:.3f}' if rho is not None else 'n/a'):>28}"
        print(line)


if __name__ == "__main__":
    main()
