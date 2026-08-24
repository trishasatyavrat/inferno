"""Correctness harness: inferno's matmul vs PyTorch's, on random data.

PyTorch is the oracle here — battle-tested by millions of users. If our
C++ agrees with it across shapes and hundreds of random trials, our
math is right. Every optimization we make later must keep this green.

Run: make pytest
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np
import torch
import inferno_core

RTOL = 1e-4  # float32 accumulation order differs between impls,
ATOL = 1e-5  # so we compare within tolerance, not bit-for-bit.

def check(m, k, n, rng):
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    ours = inferno_core.matmul(a, b)
    theirs = (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        worst = np.abs(ours - theirs).max()
        raise AssertionError(f"mismatch at ({m}x{k})@({k}x{n}): max abs diff {worst}")

def main():
    rng = np.random.default_rng(0)  # fixed seed: failures must be reproducible
    shapes = [(1, 1, 1), (2, 2, 2), (1, 7, 3), (5, 1, 5),
              (16, 16, 16), (33, 17, 9), (64, 128, 32), (128, 128, 128)]
    for m, k, n in shapes:
        check(m, k, n, rng)
    # Fuzz: 200 random shapes.
    for _ in range(200):
        m, k, n = rng.integers(1, 96, size=3)
        check(int(m), int(k), int(n), rng)
    print(f"inferno matmul matches torch on {len(shapes)} fixed + 200 random shapes")

if __name__ == "__main__":
    main()
