"""Exact random *minimal* identity sampler via BFS meet-in-the-middle.

Realises the user's observation: because the identity DB is complete up to 6
gates (and here we BFS the ball ourselves as an interim oracle), we can draw
genuinely random minimal identities of length up to ~12 on a small support,
then wire-relabel them onto the big circuit.

length L split as l1 + l2 (l1 = ceil(L/2) <= radius):
  - draw a random reduced word A of length l1 on the support,
  - its inverse function must be reachable in <= l2 = floor(L/2) gates,
  - concatenate A + B where table(B) = table(A)^{-1},
  - accept iff A+B is *minimal* (no proper contiguous sub-window is identity,
    checked exactly) -- rejection sampling gives locally-geodesic identities.
"""
import itertools
import random

import numpy as np

import g57


class MITMSampler:
    def __init__(self, support=5, radius=6, seed=None):
        self.support = support
        self.radius = radius
        self.rng = random.Random(seed)
        self.gates = list(itertools.permutations(range(support), 3))
        self.gts = {g: g57.gate_table(g, support) for g in self.gates}
        self.ID = np.arange(1 << support, dtype=np.int64)
        self._ball = None

    def _build_ball(self):
        if self._ball is not None:
            return
        ball = {self.ID.tobytes(): ()}
        frontier = {self.ID.tobytes(): ()}
        for _ in range(self.radius):
            nf = {}
            for pb, w in frontier.items():
                P = np.frombuffer(pb, dtype=np.int64)
                for g, tb in self.gts.items():
                    npb = (tb[P]).tobytes()
                    if npb not in ball:
                        ball[npb] = w + (g,)
                        nf[npb] = w + (g,)
            frontier = nf
        self._ball = ball

    def _table(self, word):
        T = self.ID.copy()
        for g in word:
            T = self.gts[g][T]
        return T

    @staticmethod
    def _minimal(word, support):
        L = len(word)
        for i in range(L):
            for j in range(i + 2, L + (1 if i > 0 else 0)):
                if j - i >= L:
                    continue
                if g57.is_identity_exact(word[i:j], support):
                    return False
        return True

    def sample(self, length, max_tries=2000):
        """A random minimal identity word of the given length on `support` wires."""
        self._build_ball()
        l1 = (length + 1) // 2
        l2 = length - l1
        if l1 > self.radius or l2 > self.radius:
            raise ValueError(f"length {length} exceeds 2*radius={2*self.radius}")
        for _ in range(max_tries):
            A = tuple(self.rng.choice(self.gates) for _ in range(l1))
            fA = self._table(A)
            inv = np.empty_like(fA)
            inv[fA] = self.ID
            B = self._ball.get(inv.tobytes())
            if B is None or len(B) != l2:
                continue
            word = list(A) + list(B)
            if not g57.is_identity_exact(word, self.support):
                continue
            if self._minimal(word, self.support):
                return word
        return None
