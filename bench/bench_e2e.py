"""End-to-end benchmark: inferno vs PyTorch on GPT-2 small (124M).

Uses weights/gpt2.bin if present, otherwise random weights at the same
shape - speed does not depend on the values. PyTorch side is the
long-form reference model from the test harness (no KV cache), plus its
raw forward pass, which is what a well-tuned CPU BLAS looks like.

Run: make bench-e2e
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import numpy as np
import torch
import inferno_core
from test_vs_torch import make_params, torch_gpt2

FULL = {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12}


def timeit(fn, reps=3):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    rng = np.random.default_rng(0)
    path = os.path.join(os.path.dirname(__file__), "..", "weights", "gpt2.bin")
    if os.path.exists(path):
        model = inferno_core.GPT2(path)
        source = "weights/gpt2.bin"
    else:
        source = "random weights, GPT-2 small shape"
        model = None
    params = make_params(FULL, rng)
    if model is None:
        model = inferno_core.GPT2(FULL["n_vocab"], FULL["n_ctx"], FULL["n_embd"], FULL["n_head"],
                                  FULL["n_layer"], params)
    print(f"model: {source}; torch {torch.__version__}, {torch.get_num_threads()} threads")

    prompt = [int(v) for v in rng.integers(0, FULL["n_vocab"], size=16)]
    seq64 = [int(v) for v in rng.integers(0, FULL["n_vocab"], size=64)]

    print("\nforward pass, 64 tokens (prefill-shaped work):")
    t_inf = timeit(lambda: model.forward(seq64))
    with torch.no_grad():
        t_torch = timeit(lambda: torch_gpt2(FULL, params, seq64))
    print(f"  inferno  {t_inf * 1e3:8.1f} ms")
    print(f"  torch    {t_torch * 1e3:8.1f} ms   (inferno is {t_torch / t_inf:.2f}x torch)")

    print("\ndecode step: 1 new token against a 64-token KV cache:")
    # 32 single-token steps after the prefill, minus the prefill alone,
    # averaged - one step is too small to time against a 350 ms prefill.
    steps = [[int(v)] for v in rng.integers(0, FULL["n_vocab"], size=32)]
    t_all = timeit(lambda: model.forward_chunked([seq64] + steps), reps=2)
    t_prefill = timeit(lambda: model.forward_chunked([seq64]), reps=2)
    per_step = max(t_all - t_prefill, 0) / len(steps)
    print(f"  inferno  {per_step * 1e3:8.1f} ms per token  "
          f"(weights are {124e6 * 4 / 1e6:.0f} MB; at this rate that is "
          f"{124e6 * 4 / per_step / 1e9:.0f} GB/s of weight traffic)")

    n_new = 32
    print(f"\ngenerate {n_new} tokens from a 16-token prompt (greedy):")
    t_cache = timeit(lambda: model.generate(prompt, max_new=n_new, temperature=0.0, use_cache=True), reps=2)
    t_nocache = timeit(lambda: model.generate(prompt, max_new=n_new, temperature=0.0, use_cache=False), reps=1)

    def torch_loop():
        toks = list(prompt)
        with torch.no_grad():
            for _ in range(n_new):
                toks.append(int(torch_gpt2(FULL, params, toks)[-1].argmax()))
        return toks
    t_torch_loop = timeit(torch_loop, reps=1)
    print(f"  inferno, KV cache       {t_cache:6.2f} s  ({n_new / t_cache:5.1f} tok/s)")
    print(f"  inferno, full recompute {t_nocache:6.2f} s  ({n_new / t_nocache:5.1f} tok/s)  "
          f"cache speedup {t_nocache / t_cache:.1f}x")
    print(f"  torch, full recompute   {t_torch_loop:6.2f} s  ({n_new / t_torch_loop:5.1f} tok/s)")


if __name__ == "__main__":
    main()
