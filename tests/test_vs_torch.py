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

# ---- A reference GPT-2, written the long way -------------------------
# Weights use the checkpoint's (in, out) layout so the same dict feeds
# both this and inferno. Random init, small configs: this tests wiring,
# not knowledge. Real weights come from the exporter later.

def make_params(cfg, rng):
    V, N, C, L = cfg["n_vocab"], cfg["n_ctx"], cfg["n_embd"], cfg["n_layer"]
    f32 = np.float32
    p = {"wte": (rng.standard_normal((V, C)) * 0.02).astype(f32),
         "wpe": (rng.standard_normal((N, C)) * 0.01).astype(f32),
         "ln_f.g": (1 + rng.standard_normal(C) * 0.1).astype(f32),
         "ln_f.b": (rng.standard_normal(C) * 0.1).astype(f32)}
    for i in range(L):
        h = f"h.{i}."
        p[h + "ln_1.g"] = (1 + rng.standard_normal(C) * 0.1).astype(f32)
        p[h + "ln_1.b"] = (rng.standard_normal(C) * 0.1).astype(f32)
        p[h + "attn.c_attn.w"] = (rng.standard_normal((C, 3 * C)) * 0.02).astype(f32)
        p[h + "attn.c_attn.b"] = (rng.standard_normal(3 * C) * 0.01).astype(f32)
        p[h + "attn.c_proj.w"] = (rng.standard_normal((C, C)) * 0.02).astype(f32)
        p[h + "attn.c_proj.b"] = (rng.standard_normal(C) * 0.01).astype(f32)
        p[h + "ln_2.g"] = (1 + rng.standard_normal(C) * 0.1).astype(f32)
        p[h + "ln_2.b"] = (rng.standard_normal(C) * 0.1).astype(f32)
        p[h + "mlp.c_fc.w"] = (rng.standard_normal((C, 4 * C)) * 0.02).astype(f32)
        p[h + "mlp.c_fc.b"] = (rng.standard_normal(4 * C) * 0.01).astype(f32)
        p[h + "mlp.c_proj.w"] = (rng.standard_normal((4 * C, C)) * 0.02).astype(f32)
        p[h + "mlp.c_proj.b"] = (rng.standard_normal(C) * 0.01).astype(f32)
    return p

def torch_gpt2(cfg, p, tokens):
    t = {k: torch.from_numpy(v) for k, v in p.items()}
    ln = lambda x, g, b: torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b, eps=1e-5)
    T = len(tokens)
    x = t["wte"][tokens] + t["wpe"][:T]
    for i in range(cfg["n_layer"]):
        h = f"h.{i}."
        a = torch_attention(ln(x, t[h + "ln_1.g"], t[h + "ln_1.b"]),
                            t[h + "attn.c_attn.w"], t[h + "attn.c_attn.b"],
                            t[h + "attn.c_proj.w"], t[h + "attn.c_proj.b"], cfg["n_head"])
        x = x + a
        m = ln(x, t[h + "ln_2.g"], t[h + "ln_2.b"]) @ t[h + "mlp.c_fc.w"] + t[h + "mlp.c_fc.b"]
        m = torch.nn.functional.gelu(m, approximate="tanh") @ t[h + "mlp.c_proj.w"] + t[h + "mlp.c_proj.b"]
        x = x + m
    x = ln(x, t["ln_f.g"], t["ln_f.b"])
    return (x @ t["wte"].T).numpy()

def check_gpt2(cfg, p, tokens, rtol=2e-3, atol=2e-3):
    model = inferno_core.GPT2(cfg["n_vocab"], cfg["n_ctx"], cfg["n_embd"], cfg["n_head"],
                              cfg["n_layer"], p)
    ours = model.forward(tokens)
    theirs = torch_gpt2(cfg, p, tokens)
    if ours.shape != theirs.shape:
        raise AssertionError(f"gpt2 logits shape {ours.shape} != {theirs.shape}")
    if not np.allclose(ours, theirs, rtol=rtol, atol=atol):
        raise AssertionError(f"gpt2 mismatch T={len(tokens)} cfg={cfg}: "
                             f"max diff {np.abs(ours - theirs).max()}")
    # The prediction that matters is the argmax; it must agree exactly.
    if not np.array_equal(ours.argmax(-1), theirs.argmax(-1)):
        raise AssertionError("gpt2 argmax disagrees")

def check_sampling(cfg, p, rng):
    model = inferno_core.GPT2(cfg["n_vocab"], cfg["n_ctx"], cfg["n_embd"], cfg["n_head"],
                              cfg["n_layer"], p)
    prompt = [7, 3, 9]

    # Greedy generation must equal "run forward, take argmax, append" -
    # the definition, spelled out in Python, against the C++ loop.
    greedy = model.generate(prompt, max_new=6, temperature=0.0)
    manual = list(prompt)
    for _ in range(6):
        manual.append(int(model.forward(manual)[-1].argmax()))
    if greedy != manual:
        raise AssertionError(f"greedy generate {greedy} != argmax loop {manual}")

    # Same seed, same output; different seed, (almost surely) different.
    a = model.generate(prompt, max_new=8, temperature=1.0, top_k=0, seed=5)
    b = model.generate(prompt, max_new=8, temperature=1.0, top_k=0, seed=5)
    c = model.generate(prompt, max_new=8, temperature=1.0, top_k=0, seed=6)
    if a != b:
        raise AssertionError("same seed produced different output")
    if a == c:
        raise AssertionError("different seeds produced identical 8-token output (suspicious)")
    if len(a) != len(prompt) + 8 or not all(0 <= t < cfg["n_vocab"] for t in a):
        raise AssertionError("generated ids out of range or wrong length")

    # top_k=1 is greedy by another name.
    if model.generate(prompt, max_new=6, temperature=1.0, top_k=1, seed=3) != greedy:
        raise AssertionError("top_k=1 should equal greedy")

    # Sampling respects top-k: over many draws from one logits row, every
    # chosen id is in the k highest. And temperature: a very cold draw is
    # nearly always the argmax; a hot one spreads out.
    logits = rng.standard_normal(cfg["n_vocab"], dtype=np.float32) * 2
    top5 = set(np.argsort(-logits)[:5].tolist())
    draws = [inferno_core.sample_next(logits, temperature=1.0, top_k=5, seed=s) for s in range(1, 400)]
    if not set(draws) <= top5:
        raise AssertionError("top-k sampling drew a token outside the top k")
    if len(set(draws)) < 3:
        raise AssertionError("top-k=5 at T=1 should hit several of the five over 400 draws")
    cold = [inferno_core.sample_next(logits, temperature=0.05, top_k=0, seed=s) for s in range(1, 200)]
    hot = [inferno_core.sample_next(logits, temperature=3.0, top_k=0, seed=s) for s in range(1, 200)]
    if len(set(cold)) > 2 or len(set(hot)) < 10:
        raise AssertionError(f"temperature not behaving: cold={len(set(cold))} distinct, hot={len(set(hot))}")

    # The context window is a hard stop: never more than n_ctx tokens.
    long_out = model.generate(prompt, max_new=100, temperature=0.0)
    if len(long_out) != cfg["n_ctx"]:
        raise AssertionError(f"generate exceeded/undershot n_ctx: {len(long_out)} vs {cfg['n_ctx']}")

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

    x = rng.standard_normal((6, 16), dtype=np.float32)
    w1 = rng.standard_normal((16, 64), dtype=np.float32) * 0.25
    b1 = rng.standard_normal(64, dtype=np.float32) * 0.1
    w2 = rng.standard_normal((64, 16), dtype=np.float32) * 0.125
    b2 = rng.standard_normal(16, dtype=np.float32) * 0.1
    ref = torch.nn.functional.gelu(torch.from_numpy(x @ w1 + b1), approximate="tanh").numpy() @ w2 + b2
    if not np.allclose(inferno_core.mlp(x, w1, b1, w2, b2), ref, rtol=1e-3, atol=1e-4):
        raise AssertionError("mlp mismatch")

    small = {"n_vocab": 100, "n_ctx": 16, "n_embd": 32, "n_head": 4, "n_layer": 2}
    p_small = make_params(small, rng)
    for T in (1, 5, 16):
        check_gpt2(small, p_small, [int(v) for v in rng.integers(0, 100, size=T)])
    # Full GPT-2 small shape at random init: 124M parameters, the real
    # dimensions the loader will fill. Slow-ish (~1 s), worth it.
    full = {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12}
    p_full = make_params(full, rng)
    check_gpt2(full, p_full, [int(v) for v in rng.integers(0, 50257, size=8)])
    print("gpt2 forward matches torch reference (2-layer toy at T=1/5/16; full 124M config at T=8)")

    # Checkpoint round-trip: write the toy params with the exporter's
    # writer, load them through the C++ loader, and the logits must be
    # bit-for-bit identical to the dict-built model - same weights, same
    # code, so any difference would mean the file format lost something.
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from export_gpt2 import write_inferno
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "toy.bin")
        write_inferno(path, small, p_small)
        loaded = inferno_core.GPT2(path)
        built = inferno_core.GPT2(small["n_vocab"], small["n_ctx"], small["n_embd"],
                                  small["n_head"], small["n_layer"], p_small)
        toks = [3, 1, 4, 1, 5, 9, 2, 6]
        if not np.array_equal(loaded.forward(toks), built.forward(toks)):
            raise AssertionError("checkpoint round-trip changed the logits")
        expected = sum(v.size for v in p_small.values())
        if loaded.n_params() != expected:
            raise AssertionError(f"n_params {loaded.n_params()} != {expected}")
    print("checkpoint round-trip: loader reproduces dict-built model bit-for-bit")

    check_sampling(small, p_small, rng)
    print("generation: greedy == argmax loop, seeds reproduce, top-k respected, "
          "temperature sharpens, n_ctx honored")

if __name__ == "__main__":
    main()
