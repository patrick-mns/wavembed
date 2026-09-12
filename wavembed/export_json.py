"""
Exports one or more checkpoints to JSON — feeds the interactive pages in
web/, which recompute the exact same harmonic formula in JavaScript, in
the browser, over the real trained weights. Each output file carries its
own `orders`/`max_amplitude` alongside the per-word numbers, so a page
can load any checkpoint (different harmonic count, different corpus) and
stay correct without hardcoding those constants in JS.

Usage:
    # single checkpoint
    python -m wavembed.export_json --checkpoint checkpoints/wavembed.pt --out web/data/ag_news_6.json

    # every known checkpoint that exists in checkpoints/, plus web/models.json
    python -m wavembed.export_json --all
"""

from __future__ import annotations

import argparse
import json
import os

import torch

# checkpoint filename -> (output name, label shown in the model picker)
KNOWN_CHECKPOINTS = {
    "wavembed.pt": ("ag_news_6", "AG News · 6 numbers/word"),
    "wavembed_text8.pt": ("text8_6", "text8 · 6 numbers/word"),
    "wavembed_text8_12.pt": ("text8_12", "text8 · 12 numbers/word"),
    "wavembed_text8_24.pt": ("text8_24", "text8 · 24 numbers/word"),
    "wavembed_text8_48.pt": ("text8_48", "text8 · 48 numbers/word"),
}


def export_one(checkpoint: str, out: str) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    params = ckpt["state_dict"]["params"]

    words = {}
    for word, idx in vocab.items():
        if word in ("<pad>", "<unk>"):
            continue
        words[word] = [round(x, 4) for x in params[idx].tolist()]

    payload = {"orders": list(ckpt["orders"]), "max_amplitude": ckpt["max_amplitude"], "words": words}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"{len(words)} words, {len(payload['orders'])} harmonics -> {out}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wavembed.pt")
    parser.add_argument("--out", type=str, default="web/vocab_embeddings.json")
    parser.add_argument("--all", action="store_true",
                         help="export every checkpoint in KNOWN_CHECKPOINTS that exists on disk, "
                              "into web/data/, and (re)write web/models.json for the model picker")
    args = parser.parse_args()

    if not args.all:
        export_one(args.checkpoint, args.out)
        return

    manifest = []
    for filename, (name, label) in KNOWN_CHECKPOINTS.items():
        path = os.path.join("checkpoints", filename)
        if not os.path.exists(path):
            print(f"skipping {filename} (not found)")
            continue
        export_one(path, f"web/data/{name}.json")
        manifest.append({"name": name, "label": label, "file": f"data/{name}.json"})

    with open("web/models.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"models.json: {len(manifest)} models")


if __name__ == "__main__":
    main()
