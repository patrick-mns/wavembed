"""
Takes a short sentence and overlays every word's wave on the SAME polar
plot, each in its own color — how a whole phrase looks at once, not one
word at a time.

Usage:
    python -m wavembed.sentence --sentence "stock market shares rose today"
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from .corpus import tokenize
from .model import WaveEmbedding
from .visualize import BG, GRID_LINE, TEXT

PALETTE = ["#f0a04b", "#5dcaa5", "#85b7eb", "#fac775", "#e08a8a",
           "#afa9ec", "#7ec8d8", "#c9a0e0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--sentence", type=str, default="stock market shares rose today")
    parser.add_argument("--out", type=str, default="media/sentence_wave.png")
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
    theta = theta.numpy()
    bump = bump.numpy()

    fig = plt.figure(figsize=(8, 8.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_facecolor(BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines["polar"].set_color(GRID_LINE)
    ax.grid(color=GRID_LINE)
    ax.set_yticklabels([])

    for i, (word, b) in enumerate(zip(found, bump)):
        color = PALETTE[i % len(PALETTE)]
        r = b - b.min() + 0.3
        ax.plot(theta, r, color=color, linewidth=2.2, label=word)
        ax.fill(theta, r, color=color, alpha=0.06)
        peak = r.argmax()
        ax.annotate(word, (theta[peak], r[peak]), color=color, fontsize=11,
                    family="monospace", fontweight="bold", ha="center")

    ax.set_title(f'"{args.sentence}"\nevery word\'s wave, overlaid on the same plot',
                 color=TEXT, fontsize=12, family="monospace", pad=24)
    fig.savefig(args.out, facecolor=BG, dpi=150, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
