"""
Same wave, unrolled: instead of a polar plot, draws energy(θ) as a plain
horizontal curve (angle on the x-axis, amplitude on the y-axis). Same
words, same colors as the polar overlay — just a different projection of
the exact same function, useful for reading off the raw shape without the
circular wrap.

Usage:
    python -m wavembed.unrolled --sentence "google war olympic president"
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from .corpus import tokenize
from .model import WaveEmbedding
from .visualize import BG, GRID_LINE, TEXT
from .sentence import PALETTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--sentence", type=str, default="google war olympic president")
    parser.add_argument("--out", type=str, default="media/unrolled_wave.png")
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
    theta_deg = np.degrees(theta.numpy())
    bump = bump.numpy()

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, alpha=0.5)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("θ (degrees)", color=TEXT, fontsize=10, family="monospace")
    ax.set_ylabel("energy(θ)", color=TEXT, fontsize=10, family="monospace")

    for i, (word, b) in enumerate(zip(found, bump)):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(theta_deg, b, color=color, linewidth=2.2, label=word)
        ax.fill_between(theta_deg, b, b.min(), color=color, alpha=0.06)

    legend = ax.legend(facecolor=BG, edgecolor=GRID_LINE, labelcolor=TEXT,
                        prop={"family": "monospace", "size": 10})

    ax.set_title(f'"{args.sentence}"\nsame waves, unrolled — no polar wraparound',
                 color=TEXT, fontsize=12, family="monospace", pad=16)
    fig.savefig(args.out, facecolor=BG, dpi=150, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
