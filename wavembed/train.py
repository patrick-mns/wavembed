"""
Trains a WaveEmbedding via skip-gram + negative sampling — no grid, no
routing, no labels: just real text, real co-occurrence. The "similarity"
used during training is the wave correlation (dot product in phasor
space, see model.py).

Usage:
    python -m wavembed.train
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .corpus import build_corpus, build_negative_sampler, sample_negatives
from .model import WaveEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--n-docs", type=int, default=60000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--n-negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=5.0,
                         help="fixed scale applied to cosine similarity before the sigmoid")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="checkpoints/wavembed.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("building skip-gram pairs from real AG News headlines...")
    vocab, targets, contexts = build_corpus(vocab_size=args.vocab_size, n_docs=args.n_docs, window=args.window)
    print(f"vocabulary: {len(vocab)} | (target, context) pairs: {targets.shape[0]}")

    neg_dist = build_negative_sampler(vocab, targets)

    model = WaveEmbedding(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    n = targets.shape[0]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            t_ids = targets[idx].to(device)
            c_ids = contexts[idx].to(device)
            neg_ids = sample_negatives(neg_dist, len(idx), args.n_negatives).to(device)  # (B, K)

            pos_sim = model.similarity(t_ids, c_ids) * args.temperature
            neg_sim = model.similarity(t_ids.unsqueeze(1).expand(-1, args.n_negatives), neg_ids) * args.temperature

            pos_loss = bce(pos_sim, torch.ones_like(pos_sim))
            neg_loss = bce(neg_sim, torch.zeros_like(neg_sim))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"  epoch {epoch:2d} | loss {total_loss/n_batches:.4f}")

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntime: {elapsed:.1f}s | parameters: {n_params}")

    import os
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "orders": model.orders, "max_amplitude": model.max_amplitude},
               args.save)
    print(f"checkpoint saved to {args.save}")


if __name__ == "__main__":
    main()
