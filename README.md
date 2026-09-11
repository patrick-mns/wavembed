# wavembed

A word embedding whose representation is a **wave**, not a vector.

Same training objective as word2vec — skip-gram with negative sampling,
learned purely from real text co-occurrence, no labels, no categories.
The difference is what each word actually *is*: instead of a point in an
abstract N-dimensional space, a word is 3 (amplitude, phase) pairs — one
per even harmonic order (2, 4, 6) — that a fixed formula turns into a
periodic function of angle:

```
energy(θ) = exp( Σ amplitude_k · cos(order_k · θ − phase_k) )
```

That function is literally drawable: plotted in polar coordinates, each
word becomes a distinctive "flower" shape. Two related words end up with
visibly similar shapes, because they're trained to.

<p align="center"><img src="media/token_waves.png" width="900"></p>

Unrelated words, by contrast, end up with visibly distinct shapes — no
two of these four share a lobe count or orientation:

<p align="center"><img src="media/contrasting_words.png" width="900"></p>

A whole sentence at once, same idea: every word's wave drawn on the same
polar plot, each in its own color. Words from the same topic land as
near-identical, overlapping petals; the odd one out stands apart:

<p align="center"><img src="media/sentence_wave.png" width="500"></p>

Same function, unrolled: angle on the x-axis instead of wrapped around a
circle. No new information over the polar view — just easier to read the
raw shape and compare peaks side by side:

<p align="center"><img src="media/unrolled_wave.png" width="900"></p>

## Why even harmonics, specifically

An even-order term satisfies `f(θ + π) = f(θ)` — the shape is symmetric
under inversion. That guarantees the wave's center of mass never moves,
no matter what the training finds: a word's identity is carried entirely
by *the shape* (which directions it bulges toward), never by a position.
This is what makes every word directly comparable and drawable on the
same footing.

## Why this beats plain cosine similarity for comparison

Harmonics of different orders are **orthogonal** over a full period.
That means the correlation between two words' waves has an exact closed
form — no numerical integration needed. Convert each harmonic's
`(amplitude, phase)` into Cartesian coordinates
`(amplitude·cos(phase), amplitude·sin(phase))` (call this the *phasor*
representation), and the dot product of two words' phasor vectors **is**
their wave correlation. `model.py`'s `similarity()` does exactly this.

## Why wave, not vector, at all

- **Sum = interference, for free.** Adding two vectors is just
  arithmetic — it doesn't know if they agree. Adding two waves produces
  constructive or destructive interference automatically, by the physics
  of the sum. Agreement is built into the operation, not computed after
  the fact.
- **Independent channels by frequency.** Each harmonic order is
  orthogonal to the others — you can read or inject information at one
  frequency without touching the rest, like frequency-division
  multiplexing. A matrix has no such separation unless you engineer block
  structure by hand.
- **Physically realizable outside digital silicon.** Wave addition is
  physical superposition — optical, acoustic, RF interference happens
  passively, at the speed of light, no multiply-accumulate required. This
  is a real door into analog/optical/neuromorphic computing that matrix
  multiplication doesn't open without converting everything back to
  digital multiply-adds.
- **Multiplying (not just summing) waves shifts frequency.** That's a
  second, different composition operator — closer to *binding* two
  concepts (AM/FM-style modulation) than to *blending* them (a weighted
  average). A matrix only has one natural combination operator (weighted
  sum); a wave has two, with different meanings.

**Subtraction makes the interference argument concrete.** Subtracting one
word's wave from another's isolates what's left after their shared shape
cancels out — and how much survives tracks correlation directly:

| pair | correlation | amplitude of A − B |
|---|---|---|
| `stock` − `market` (near-identical) | 0.988 | **20%** of either original — most of the shape cancels |
| `google` − `war` (unrelated, slightly anti-correlated) | -0.241 | **181%** of either original — nothing cancels, it reinforces instead |

This isn't tuned to look nice — it's a direct consequence of subtracting
two functions built from the same orthogonal basis: the more two waves
agree, the more they cancel when subtracted; the more they disagree
(especially when negatively correlated), the more the difference grows
past either input alone.

## Results

Trained on 60k real AG News headlines (~15.8M skip-gram pairs), 8000-word
vocabulary, 25 epochs. Word pairs, wave correlation:

| pair | similarity |
|---|---|
| `stock` ↔ `shares` | **0.994** |
| `team` ↔ `championship` | **0.987** |
| `google` ↔ `microsoft` | **0.981** |
| `google` ↔ `yahoo` | **0.990** |
| `microsoft` ↔ `software` | **0.995** |
| `google` ↔ `war` (unrelated) | **-0.241** |
| `stock` ↔ `olympic` (unrelated) | **-0.133** |
| `president` ↔ `basketball` (unrelated) | **0.030** |

Every related pair lands strongly positive; every unrelated pair lands
near zero or negative. First attempt (higher learning rate, fewer epochs)
partially collapsed — some unrelated pairs spiked to spuriously high
similarity because training got stuck early. Lowering the learning rate
and training longer fixed it; see `train.py`'s defaults.

## Usage

```bash
pip install -r requirements.txt

# train from scratch (downloads AG News via Hugging Face `datasets`)
python -m wavembed.train

# check similarity between word pairs
python -m wavembed.similarity

# draw each token of a sentence as its own wave
python -m wavembed.visualize --sentence "the stock market fell sharply today"

# overlay every word of a sentence on the same polar plot, one color each
python -m wavembed.sentence --sentence "stock market shares rose today"

# same idea, unrolled onto a plain x/y plot instead of polar
python -m wavembed.unrolled --sentence "google war olympic president"
```

For your own corpus instead of AG News, use `corpus.build_corpus_from_texts(texts)`.

## Interactive demos

Both pages are self-contained (no server, no backend) and recompute the
exact same harmonic formula in JavaScript, in the browser, over the real
trained weights.

- `web/wave_tokenizer.html` — type a sentence and watch it split into
  tokens, each one its own live wave, in polar or unrolled-horizontal
  view.
- `web/wave_calculator.html` — pick two words and watch the four basic
  operations act on their waves: **sum** (interference — agreement
  reinforces, disagreement cancels), **difference** (isolates what's in A
  but not B — two nearly-identical words like `stock`/`market` collapse
  to a residual only ~20% the size of either original), **product**
  (frequency mixing), **quotient** (spikes visibly wherever B is near
  zero).

Regenerate their embedded data with:

```bash
python -m wavembed.export_json
```

## Related work — where this fits

word2vec (2013) is old. This project doesn't pretend otherwise, and isn't
trying to compete with what's actually used in production today:

- **word2vec (2013) / GloVe (2014)** — static embeddings: one fixed vector
  per word, regardless of context. This is the era `wavembed` borrows its
  training objective from (skip-gram + negative sampling).
- **fastText (2016)** — same static idea, extended with subword
  (character n-gram) information, better on rare/unseen words.
- **ELMo (2018) → BERT and successors** — the real shift: *contextual*
  embeddings. The same word gets a different vector depending on the
  sentence it's in ("bank" of a river vs. a "bank" account), computed by
  a Transformer rather than looked up in a fixed table.
- **Today** — production-grade text embeddings (OpenAI/Cohere/Google
  embedding APIs, open models like E5, BGE, GTE) are built on large
  pretrained Transformers, contextual, hundreds to thousands of
  dimensions, trained on web-scale data with contrastive objectives far
  more sophisticated than plain negative sampling.

`wavembed` sits deliberately outside that progression. It isn't a step
toward beating E5 or BGE — 6 numbers per word, a single small corpus, and
no context-sensitivity couldn't plausibly compete on raw quality with
models trained on billions of tokens. The actual question being tested is
orthogonal to "which embedding is best today": **does swapping the
*representation* of a classical, well-understood objective (skip-gram)
from an arbitrary vector to a physically interpretable wave preserve the
useful properties of embeddings (real relational structure, meaningful
similarity, analogy) while adding new ones (drawability, exact
closed-form correlation, physical/analog realizability)?** word2vec is
used here as a controlled, minimal, easy-to-reason-about baseline for
that question — not as a target to surpass.

## Status

This is an early, deliberately closed experiment (~8000-word vocabulary,
6-number-per-word capacity, single corpus) meant to validate the core
idea before scaling it into a proper paper. Known open questions:
representational capacity at larger vocabularies, comparison against
standard word2vec/GloVe on established benchmarks (e.g. WordSim-353), and
formal evaluation beyond spot-checked word pairs.
