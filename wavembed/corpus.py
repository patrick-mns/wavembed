"""
Builds skip-gram (target, context) pairs + negative sampling straight from
real text — the same recipe as the original word2vec. No labels, no
categories, just text.

Standalone: this module doesn't depend on any other project. Point it at
any list of strings (`build_corpus_from_texts`) or use `build_corpus` for
a quick AG News-backed corpus (real news headlines, convenient and free).
"""

from __future__ import annotations

import re
from collections import Counter

import torch

PAD = "<pad>"
UNK = "<unk>"
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab = {PAD: 0, UNK: 1}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def build_corpus_from_texts(texts: list[str], vocab_size: int = 8000, window: int = 4):
    """Devolve (vocab, targets, contexts) a partir de uma lista de strings."""
    vocab = build_vocab(texts, vocab_size)
    unk = vocab[UNK]

    targets, contexts = [], []
    for text in texts:
        ids = [vocab.get(w, unk) for w in tokenize(text)]
        ids = [i for i in ids if i != unk]
        n = len(ids)
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                targets.append(ids[i])
                contexts.append(ids[j])

    return vocab, torch.tensor(targets, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


def build_corpus(vocab_size: int = 8000, n_docs: int = 60000, window: int = 4, seed: int = 0):
    """Convenience: pulls real AG News headlines (via HF `datasets`) as a
    free, ready-made text corpus. Swap for your own corpus with
    `build_corpus_from_texts` if you don't want this dependency."""
    from datasets import load_dataset

    ds = load_dataset("ag_news")["train"].shuffle(seed=seed).select(range(n_docs))
    return build_corpus_from_texts(ds["text"], vocab_size=vocab_size, window=window)


def build_negative_sampler(vocab: dict, targets: torch.Tensor, power: float = 0.75) -> torch.Tensor:
    """Unigram distribution raised to 0.75 (the word2vec trick) — frequent
    words get sampled as negatives more often, but not proportionally."""
    counts = torch.bincount(targets, minlength=len(vocab)).float()
    counts = counts.clamp(min=1) ** power
    return counts / counts.sum()


def sample_negatives(dist: torch.Tensor, n: int, k: int, generator=None) -> torch.Tensor:
    """n examples, k negatives each -> (n, k)."""
    flat = torch.multinomial(dist, n * k, replacement=True, generator=generator)
    return flat.view(n, k)
