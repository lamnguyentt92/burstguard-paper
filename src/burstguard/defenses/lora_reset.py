"""Periodic partial LoRA reset, adapted from Vuillod et al. (FPS 2025, arXiv:2511.14406).

Every `every` rounds, a random `frac` of the columns of each LoRA A matrix and of the rows of each LoRA B matrix
are reset to their initial values; reset indices are not selected again for `cooldown` rounds. The original work
re-initialises with PiSSA; here the adapters use the default LoRA initialisation (A random, B = 0), so they are
reset to that initialisation.
"""
from __future__ import annotations

import numpy as np


class LoraReset:
    def __init__(self, segments, init_vec: np.ndarray, every: int, frac: float, cooldown: int, seed: int = 0):
        self.segments = segments
        self.init = init_vec.copy()
        self.every, self.frac, self.cooldown = int(every), float(frac), int(cooldown)
        self.rng = np.random.default_rng(30_000 + seed)
        self.last_reset: dict[tuple[str, int], int] = {}
        if not segments:
            raise ValueError("lora_reset: no LoRA matrices found in the trainable parameters")

    def apply(self, theta: np.ndarray, rnd: int) -> tuple[np.ndarray, dict]:
        if self.every <= 0 or rnd % self.every != 0:
            return theta, {"reset": 0}
        theta = theta.copy()
        count = 0
        for name, off, size, shape in self.segments:
            M = theta[off:off + size].reshape(shape)
            I = self.init[off:off + size].reshape(shape)
            is_a = ".lora_A." in name
            dim = shape[1] if is_a else shape[0]          # columns of A (in_features), rows of B (out_features)
            k = max(1, int(round(self.frac * dim)))
            candidates = [j for j in range(dim) if rnd - self.last_reset.get((name, j), -10**9) > self.cooldown]
            if not candidates:
                continue
            pick = self.rng.choice(candidates, size=min(k, len(candidates)), replace=False)
            for j in pick:
                if is_a:
                    M[:, j] = I[:, j]
                else:
                    M[j, :] = I[j, :]
                self.last_reset[(name, int(j))] = rnd
                count += 1
            theta[off:off + size] = M.ravel()
        return theta, {"reset": count}
