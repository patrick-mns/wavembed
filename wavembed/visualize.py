"""
Takes a sentence and draws each token's wave separately — one polar panel
per word, in the order they appear.

Usage:
    python -m wavembed.visualize --checkpoint checkpoints/wavembed.pt \
        --sentence "the stock market fell sharply today"
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from .corpus import tokenize
from .model import WaveEmbedding

BG = "#141410"
ENERGY = "#f0a04b"
TEXT = "#c9c7bd"
GRID_LINE = "#3a3833"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--sentence", type=str, default="the stock market fell sharply today")
    parser.add_argument("--out", type=str, default="media/token_waves.png")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    model = WaveEmbedding(len(vocab), orders=ckpt["orders"], max_amplitude=ckpt["max_amplitude"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    words = tokenize(args.sentence)
    found = [w for w in words if w in vocab]
    missing = [w for w in words if w not in vocab]
    if missing:
        print(f"out of vocabulary (skipped): {missing}")
    if not found:
        print("none of the sentence's words are in the trained vocabulary")
        return

    ids = torch.tensor([vocab[w] for w in found])
    with torch.no_grad():
        theta, bump = model.profile(ids)  # bump: (n_words, n_points)
    bump = bump.numpy()
    theta = theta.numpy()

    fig, axes = plt.subplots(1, len(found), figsize=(3.6 * len(found), 4.6),
                              subplot_kw={"projection": "polar"})
    if len(found) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    for ax, word, b in zip(axes, found, bump):
        ax.set_facecolor(BG)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.tick_params(colors=TEXT, labelsize=6)
        ax.spines["polar"].set_color(GRID_LINE)
        ax.grid(color=GRID_LINE)
        ax.set_yticklabels([])
        r = b - b.min() + 0.3
        ax.plot(theta, r, color=ENERGY, linewidth=2)
        ax.fill(theta, r, color=ENERGY, alpha=0.08)
        ax.set_title(word, color=TEXT, fontsize=12, family="monospace", pad=14)

    fig.suptitle(f'"{args.sentence}"  —  each token\'s wave, learned only from text co-occurrence',
                 color=TEXT, fontsize=11.5, family="monospace", y=1.04)
    fig.savefig(args.out, facecolor=BG, dpi=150, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
