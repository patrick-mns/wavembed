"""
wave2vec — a word embedding whose representation is a wave, not a vector.

Each token gets a small table of 6 numbers: 3 (amplitude, phase) pairs,
one per harmonic order (2, 4, 6). A fixed formula turns those 6 numbers
into a periodic function of angle theta:

    energy(theta) = exp( sum_k amplitude_k * cos(order_k * theta - phase_k) )

Only EVEN harmonic orders are used. This is a deliberate constraint, not
an implementation detail: an even-order term satisfies f(theta + pi) =
f(theta), i.e. the function is point-symmetric. That guarantees the shape
never leaks information into "where" it is centered — its center of mass
is fixed by construction, so the identity of a token is carried entirely
by the *shape* of the wave (which directions it bulges toward), never by
a position. That's what makes each token drawable as a distinctive polar
"flower" and directly comparable to any other token's flower.

Similarity between two tokens is the correlation of their waves. Because
harmonics of different orders are orthogonal over a full period, that
correlation has an exact closed form: convert each harmonic's
(amplitude, phase) into Cartesian coordinates
(amplitude*cos(phase), amplitude*sin(phase)) — call this the "phasor"
representation — and the dot product of two tokens' phasor vectors *is*
their wave correlation, with no numerical integration required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ORDERS = (2, 4, 6)
MAX_AMPLITUDE = 2.5


class WaveEmbedding(nn.Module):
    def __init__(self, vocab_size: int, orders: tuple[int, ...] = ORDERS, max_amplitude: float = MAX_AMPLITUDE):
        super().__init__()
        self.orders = orders
        self.max_amplitude = max_amplitude
        self.k = len(orders)
        # raw (amplitude, phase) per harmonic, per token
        self.params = nn.Parameter(torch.randn(vocab_size, 2 * self.k) * 0.5)

    def amp_phase(self, ids: torch.Tensor):
        """ids: (...,) -> amplitude (..., K), phase (..., K)"""
        p = self.params[ids].view(*ids.shape, self.k, 2)
        amp = torch.sigmoid(p[..., 0]) * self.max_amplitude
        phase = p[..., 1]
        return amp, phase

    def phasor(self, ids: torch.Tensor) -> torch.Tensor:
        """Cartesian form of each harmonic: (amp*cos(phase), amp*sin(phase)).
        The dot product of two tokens' phasor vectors equals their true
        wave correlation (harmonics of different orders are orthogonal,
        so the correlation decomposes exactly into this sum)."""
        amp, phase = self.amp_phase(ids)
        return torch.cat([amp * torch.cos(phase), amp * torch.sin(phase)], dim=-1)  # (..., 2K)

    def similarity(self, ids_a: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in phasor space = normalized wave correlation."""
        pa, pb = self.phasor(ids_a), self.phasor(ids_b)
        return torch.nn.functional.cosine_similarity(pa, pb, dim=-1)

    def profile(self, ids: torch.Tensor, n_points: int = 181):
        """Angular profile for drawing: (theta, bump), bump shaped (..., n_points)."""
        amp, phase = self.amp_phase(ids)  # (..., K)
        theta = torch.linspace(0, 2 * torch.pi, n_points, device=ids.device)  # (n_points,)
        bump = torch.zeros(*ids.shape, n_points, device=ids.device)
        for k, order in enumerate(self.orders):
            bump = bump + amp[..., k].unsqueeze(-1) * torch.cos(order * theta - phase[..., k].unsqueeze(-1))
        return theta, bump
