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

from .corpus import build_corpus, build_corpus_text8, build_negative_sampler, sample_negatives
from .model import WaveEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="ag_news", choices=["ag_news", "text8"],
                         help="ag_news: ~60k headlines (small, fast). text8: ~17M tokens of "
                              "Wikipedia text (the classic word2vec benchmark corpus, much bigger).")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--n-docs", type=int, default=60000, help="ag_news only")
    parser.add_argument("--max-words", type=int, default=None, help="text8 only: cap on tokens used")
    parser.add_argument("--orders", type=str, default="2,4,6",
                         help="comma-separated even harmonic orders, e.g. '2,4,6,8,10,12' doubles "
                              "capacity to 12 numbers/word instead of the default 6")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--n-negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192,
                         help="the model is tiny (a few hundred thousand params) so with small "
                              "batches the Python/optimizer-step overhead per iteration dominates, "
                              "not the actual math — a big batch cuts wall-clock time a lot for "
                              "free, with no real effect on convergence at this scale")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=5.0,
                         help="fixed scale applied to cosine similarity before the sigmoid")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                         help="'auto' picks mps (Apple GPU) or cuda if available, else cpu")
    parser.add_argument("--save", type=str, default="checkpoints/wavembed.pt")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            resolved = "mps"
        elif torch.cuda.is_available():
            resolved = "cuda"
        else:
            resolved = "cpu"
    else:
        resolved = args.device
    device = torch.device(resolved)
    print(f"device: {resolved}")

    if args.corpus == "text8":
        print("building skip-gram pairs from text8...")
        vocab, targets, contexts = build_corpus_text8(vocab_size=args.vocab_size, window=args.window,
                                                        max_words=args.max_words)
    else:
        print("building skip-gram pairs from real AG News headlines...")
        vocab, targets, contexts = build_corpus(vocab_size=args.vocab_size, n_docs=args.n_docs, window=args.window)
    print(f"vocabulary: {len(vocab)} | (target, context) pairs: {targets.shape[0]}")

    neg_dist = build_negative_sampler(vocab, targets)

    orders = tuple(int(o) for o in args.orders.split(","))
    model = WaveEmbedding(len(vocab), orders=orders).to(device)
    print(f"harmonics: {orders} ({2 * len(orders)} numbers/word)")
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
