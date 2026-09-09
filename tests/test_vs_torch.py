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

def check_layernorm(rows, cols, rng):
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    g = rng.standard_normal(cols, dtype=np.float32)
    b = rng.standard_normal(cols, dtype=np.float32)
    ours = inferno_core.layernorm(x, g, b)
    theirs = torch.nn.functional.layer_norm(
        torch.from_numpy(x), (cols,), torch.from_numpy(g), torch.from_numpy(b), eps=1e-5
    ).numpy()
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"layernorm mismatch ({rows}x{cols}): "
                             f"max diff {np.abs(ours - theirs).max()}")

def check_softmax(rows, cols, rng):
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    ours = inferno_core.softmax(x)
    theirs = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"softmax mismatch ({rows}x{cols})")
    # Rows must sum to 1 - the property that makes it a distribution.
    if not np.allclose(ours.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("softmax rows do not sum to 1")

def check_softmax_stability():
    # Large logits: a naive exp(x)/sum(exp(x)) overflows to NaN here.
    # The max-subtraction in our implementation is what prevents it.
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    ours = inferno_core.softmax(x)
    if not np.all(np.isfinite(ours)):
        raise AssertionError("softmax overflowed on large logits")
    theirs = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        raise AssertionError("softmax mismatch on large logits")

def check_gelu(rows, cols, rng):
    x = rng.standard_normal((rows, cols), dtype=np.float32) * 3.0
    ours = inferno_core.gelu(x)
    # GPT-2 uses the tanh approximation, so compare against that exact
    # variant - not torch's default (erf-based) gelu, which differs
    # slightly and would make a correct implementation look wrong.
    theirs = torch.nn.functional.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"gelu mismatch: max diff {np.abs(ours - theirs).max()}")

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
    print(f"matmul  matches torch on {len(shapes)} fixed + 200 random shapes")

    op_shapes = [(1, 1), (1, 8), (4, 16), (7, 13), (32, 768), (64, 50257)]
    for r, c in op_shapes:
        check_layernorm(r, c, rng)
        check_softmax(r, c, rng)
        check_gelu(r, c, rng)
    check_softmax_stability()
    print(f"layernorm/softmax/gelu match torch on {len(op_shapes)} shapes "
          f"(+ softmax overflow guard)")

if __name__ == "__main__":
    main()
