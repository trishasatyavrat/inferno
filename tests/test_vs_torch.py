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

def torch_attention(x, w_qkv, b_qkv, w_proj, b_proj, n_head):
    """Reference causal self-attention, written the long way in PyTorch.

    This is the textbook version (nanoGPT's, with weights in GPT-2's
    (in, out) layout). If our C++ agrees with it at random weights across
    shapes, the wiring - fused QKV split, per-head slicing, scaling, the
    causal mask, concatenation, output projection - is right.
    """
    T, C = x.shape
    hs = C // n_head
    qkv = x @ w_qkv + b_qkv
    q, k, v = qkv.split(C, dim=1)
    q = q.view(T, n_head, hs).transpose(0, 1)  # (n_head, T, hs)
    k = k.view(T, n_head, hs).transpose(0, 1)
    v = v.view(T, n_head, hs).transpose(0, 1)
    att = (q @ k.transpose(-2, -1)) / (hs ** 0.5)  # (n_head, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = att.masked_fill(~mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    y = (att @ v).transpose(0, 1).reshape(T, C)
    return y @ w_proj + b_proj

def check_attention(T, C, n_head, rng):
    # Weight scale ~ 1/sqrt(C), as trained networks have; unit-scale
    # random weights would blow the softmax into one-hot territory and
    # hide bugs behind saturated outputs.
    s = float(1.0 / np.sqrt(C))
    x = rng.standard_normal((T, C), dtype=np.float32)
    w_qkv = (rng.standard_normal((C, 3 * C), dtype=np.float32) * s)
    b_qkv = (rng.standard_normal(3 * C, dtype=np.float32) * 0.1)
    w_proj = (rng.standard_normal((C, C), dtype=np.float32) * s)
    b_proj = (rng.standard_normal(C, dtype=np.float32) * 0.1)
    ours = inferno_core.attention(x, w_qkv, b_qkv, w_proj, b_proj, n_head)
    t = lambda a: torch.from_numpy(a)
    theirs = torch_attention(t(x), t(w_qkv), t(b_qkv), t(w_proj), t(b_proj), n_head).numpy()
    if not np.allclose(ours, theirs, rtol=1e-3, atol=1e-4):
        raise AssertionError(f"attention mismatch (T={T}, C={C}, heads={n_head}): "
                             f"max diff {np.abs(ours - theirs).max()}")

def check_causality(rng):
    # The property that makes it "causal": changing token j must not
    # change any output row i < j. Perturb the last token and confirm
    # every earlier row is bit-for-bit identical.
    T, C, n_head = 8, 32, 4
    s = float(1.0 / np.sqrt(C))
    x = rng.standard_normal((T, C), dtype=np.float32)
    w_qkv = rng.standard_normal((C, 3 * C), dtype=np.float32) * s
    b_qkv = np.zeros(3 * C, dtype=np.float32)
    w_proj = rng.standard_normal((C, C), dtype=np.float32) * s
    b_proj = np.zeros(C, dtype=np.float32)
    base = inferno_core.attention(x, w_qkv, b_qkv, w_proj, b_proj, n_head)
    x2 = x.copy()
    x2[-1] += 5.0
    changed = inferno_core.attention(x2, w_qkv, b_qkv, w_proj, b_proj, n_head)
    if not np.array_equal(base[:-1], changed[:-1]):
        raise AssertionError("causal mask leak: earlier rows changed when a later token did")
    if np.array_equal(base[-1], changed[-1]):
        raise AssertionError("last row should have changed")

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

    x = rng.standard_normal((5, 8), dtype=np.float32)
    w = rng.standard_normal((8, 3), dtype=np.float32)
    b = rng.standard_normal(3, dtype=np.float32)
    if not np.allclose(inferno_core.linear(x, w, b), x @ w + b, rtol=RTOL, atol=ATOL):
        raise AssertionError("linear mismatch")

    attn_shapes = [(1, 16, 1), (1, 16, 4), (5, 32, 4), (16, 64, 8),
                   (32, 768, 12), (64, 768, 12), (257, 768, 12)]
    for T, C, n_head in attn_shapes:
        check_attention(T, C, n_head, rng)
    check_causality(rng)
    print(f"attention matches torch on {len(attn_shapes)} shapes "
          f"(incl. GPT-2 small: C=768, 12 heads) + causality check")

if __name__ == "__main__":
    main()
