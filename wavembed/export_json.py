"""
Exports the embedding table of a checkpoint to JSON — feeds the
interactive page in web/, which recomputes the exact same harmonic
formula in JavaScript, in the browser, over the real trained weights.

Usage:
    python -m wavembed.export_json --checkpoint checkpoints/wavembed.pt --out web/vocab_embeddings.json
"""

from __future__ import annotations

import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--out", type=str, default="web/vocab_embeddings.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    params = ckpt["state_dict"]["params"]

    data = {}
    for word, idx in vocab.items():
        if word in ("<pad>", "<unk>"):
            continue
        data[word] = [round(x, 4) for x in params[idx].tolist()]

    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"{len(data)} words exported to {args.out}")


if __name__ == "__main__":
    main()
