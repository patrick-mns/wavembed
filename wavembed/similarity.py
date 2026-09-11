"""
Quick similarity check between word pairs, using wave correlation
(cosine similarity in phasor space — see model.py for why this is exact,
not an approximation).

Usage:
    python -m wavembed.similarity --checkpoint checkpoints/wavembed.pt \
        --pairs "stock,shares" "team,championship" "stock,olympic"
"""

from __future__ import annotations

import argparse

import torch

from .model import WaveEmbedding

DEFAULT_PAIRS = [
    ("google", "microsoft"), ("google", "yahoo"), ("microsoft", "software"),
    ("google", "war"), ("war", "military"), ("war", "iraq"),
    ("stock", "shares"), ("stock", "market"), ("stock", "olympic"),
    ("team", "championship"), ("team", "coach"), ("team", "computer"),
    ("president", "minister"), ("president", "government"), ("president", "basketball"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                         help='word pairs as "a,b" (space-separated); defaults to a built-in demo set')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    model = WaveEmbedding(len(vocab), orders=ckpt["orders"], max_amplitude=ckpt["max_amplitude"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.pairs:
        pairs = [tuple(p.split(",")) for p in args.pairs]
    else:
        pairs = DEFAULT_PAIRS

    for a, b in pairs:
        a, b = a.strip(), b.strip()
        if a not in vocab or b not in vocab:
            print(f"{a} or {b} not in vocabulary")
            continue
        ia, ib = torch.tensor([vocab[a]]), torch.tensor([vocab[b]])
        sim = model.similarity(ia, ib).item()
        print(f"{a:12s} <-> {b:12s}  similarity = {sim:+.3f}")


if __name__ == "__main__":
    main()
