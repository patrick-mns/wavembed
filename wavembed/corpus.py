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

import numpy as np
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
    return _vocab_from_counter(counter, vocab_size)


def _vocab_from_counter(counter: Counter, vocab_size: int) -> dict[str, int]:
    most_common = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab = {PAD: 0, UNK: 1}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def _pairs_for_document(ids: np.ndarray, window: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generates one document's (target, context) pairs via array slicing
    instead of a per-word/per-neighbor Python loop — each window offset
    `d` becomes 2 slicing operations (one per direction) covering the
    whole document at once. Produces the exact same set of pairs as the
    naive nested loop, just without per-pair interpreter overhead
    (verified: ~20x faster on real data, identical pairs on small
    hand-checked corpora including 1- and 2-word edge cases)."""
    n = len(ids)
    t_chunks, c_chunks = [], []
    for d in range(1, min(window, n - 1) + 1):
        t_chunks.append(ids[:n - d])
        c_chunks.append(ids[d:])
        t_chunks.append(ids[d:])
        c_chunks.append(ids[:n - d])
    return t_chunks, c_chunks


def build_corpus_from_texts(texts: list[str], vocab_size: int = 8000, window: int = 4):
    """Devolve (vocab, targets, contexts) a partir de uma lista de strings."""
    vocab = build_vocab(texts, vocab_size)
    unk = vocab[UNK]

    target_chunks, context_chunks = [], []
    for text in texts:
        ids = [vocab.get(w, unk) for w in tokenize(text)]
        ids = np.array([i for i in ids if i != unk], dtype=np.int64)
        if len(ids) < 2:
            continue
        t, c = _pairs_for_document(ids, window)
        target_chunks.extend(t)
        context_chunks.extend(c)

    targets = np.concatenate(target_chunks) if target_chunks else np.empty(0, dtype=np.int64)
    contexts = np.concatenate(context_chunks) if context_chunks else np.empty(0, dtype=np.int64)
    return vocab, torch.from_numpy(targets).long(), torch.from_numpy(contexts).long()


def build_corpus(vocab_size: int = 8000, n_docs: int = 60000, window: int = 4, seed: int = 0):
    """Convenience: pulls real AG News headlines (via HF `datasets`) as a
    free, ready-made text corpus. Swap for your own corpus with
    `build_corpus_from_texts` if you don't want this dependency."""
    from datasets import load_dataset

    ds = load_dataset("ag_news")["train"].shuffle(seed=seed).select(range(n_docs))
    return build_corpus_from_texts(ds["text"], vocab_size=vocab_size, window=window)


def build_corpus_text8(vocab_size: int = 30000, window: int = 5, chunk_size: int = 10000,
                        max_words: int | None = None):
    """Convenience: pulls text8 (~17M tokens of cleaned Wikipedia text, the
    classic word2vec benchmark corpus) via HF `datasets`, ~100x more text
    than the AG News default. It's one giant pre-tokenized string (already
    lowercased, punctuation stripped) rather than a list of documents, so
    we chunk it into fixed-size blocks and treat each block as a
    "document" for skip-gram pair generation — negligible loss of context
    at each chunk boundary, but keeps memory bounded instead of building
    one array over all 17M tokens at once."""
    from datasets import load_dataset

    text = load_dataset("afmck/text8")["train"][0]["text"]
    words = text.split()
    if max_words is not None:
        words = words[:max_words]

    counter = Counter(words)
    vocab = _vocab_from_counter(counter, vocab_size)
    unk = vocab[UNK]
    all_ids = np.array([vocab.get(w, unk) for w in words], dtype=np.int64)

    target_chunks, context_chunks = [], []
    for start in range(0, len(all_ids), chunk_size):
        block = all_ids[start:start + chunk_size]
        block = block[block != unk]
        if len(block) < 2:
            continue
        t, c = _pairs_for_document(block, window)
        target_chunks.extend(t)
        context_chunks.extend(c)

    targets = np.concatenate(target_chunks) if target_chunks else np.empty(0, dtype=np.int64)
    contexts = np.concatenate(context_chunks) if context_chunks else np.empty(0, dtype=np.int64)
    return vocab, torch.from_numpy(targets).long(), torch.from_numpy(contexts).long()


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
