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
```

For your own corpus instead of AG News, use `corpus.build_corpus_from_texts(texts)`.

## Interactive demo

`web/wave_tokenizer.html` is a self-contained page (no server, no
backend) that recomputes the exact same harmonic formula in JavaScript,
in the browser, over the real trained weights. Open it directly, type a
sentence, and watch it split into tokens — each one its own live wave, in
polar or unrolled-horizontal view. Regenerate its embedded data with:

```bash
python -m wavembed.export_json
```

## Status

This is an early, deliberately closed experiment (~8000-word vocabulary,
6-number-per-word capacity, single corpus) meant to validate the core
idea before scaling it into a proper paper. Known open questions:
representational capacity at larger vocabularies, comparison against
standard word2vec/GloVe on established benchmarks (e.g. WordSim-353), and
formal evaluation beyond spot-checked word pairs.
