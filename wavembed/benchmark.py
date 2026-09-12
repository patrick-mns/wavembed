"""
Standard word-similarity benchmarks: WordSim-353 (similarity + relatedness
subsets) and SimLex-999. Each dataset is (word1, word2, human_score) —
we compute the model's own similarity for every pair and report the
Spearman correlation against the human scores. This is the same metric
reported for word2vec/GloVe/fastText in the literature, so it's directly
comparable.

Datasets from https://github.com/vecto-ai/word-benchmarks (word1, word2,
similarity columns), stored under data/benchmarks/.

Usage:
    python -m wavembed.benchmark --checkpoint checkpoints/wavembed.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from scipy.stats import spearmanr

from .model import WaveEmbedding

DATASETS = {
    "SimLex-999": "data/benchmarks/simlex999.csv",
    "WordSim-353 (similarity)": "data/benchmarks/wordsim353-sim.csv",
    "WordSim-353 (relatedness)": "data/benchmarks/wordsim353-rel.csv",
}


def load_pairs(path: str) -> list[tuple[str, str, float]]:
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("word1") or not row.get("similarity"):
                continue
            pairs.append((row["word1"].lower(), row["word2"].lower(), float(row["similarity"])))
    return pairs


def evaluate(model: WaveEmbedding, vocab: dict, pairs: list[tuple[str, str, float]]):
    model_scores, human_scores, missing = [], [], 0
    for w1, w2, human in pairs:
        if w1 not in vocab or w2 not in vocab:
            missing += 1
            continue
        ids_a = torch.tensor([vocab[w1]])
        ids_b = torch.tensor([vocab[w2]])
        with torch.no_grad():
            sim = model.similarity(ids_a, ids_b).item()
        model_scores.append(sim)
        human_scores.append(human)
    if len(model_scores) < 2:
        return None, 0, missing
    rho, _ = spearmanr(model_scores, human_scores)
    return rho, len(model_scores), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    model = WaveEmbedding(len(vocab), orders=ckpt["orders"], max_amplitude=ckpt["max_amplitude"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"vocab size: {len(vocab)}\n")
    print(f"{'dataset':<28} {'spearman rho':>12} {'pairs covered':>16} {'skipped (OOV)':>14}")
    print("-" * 72)
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"{name:<28} {'(missing file)':>12}")
            continue
        pairs = load_pairs(path)
        rho, covered, missing = evaluate(model, vocab, pairs)
        rho_str = f"{rho:.3f}" if rho is not None else "n/a"
        print(f"{name:<28} {rho_str:>12} {covered:>16} {missing:>14}")


if __name__ == "__main__":
    main()
