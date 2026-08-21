# inferno

A GPT-2 inference engine built from scratch in C++ — no PyTorch, no ML
libraries. Loads real GPT-2 weights and generates text using tensor
math implemented by hand, then makes it fast: cache blocking, SIMD,
multithreading, with honest benchmarks against PyTorch at every step.

**Why:** every AI framework hides the same core — a few tensor
operations, mostly matrix multiplication, executed as fast as the
hardware allows. This project builds that core in the open, to
understand exactly what runs when a language model generates a word.

## Status

Early — day 1. Currently: tensor type + naive matmul with tests.
Roadmap below tracks progress.

- [x] Tensor type (float32, row-major) + naive matmul + tests
- [ ] Python bindings (pybind11) + correctness harness vs PyTorch
- [ ] Transformer layers: embedding, layernorm, attention, MLP
- [ ] Load real GPT-2 weights → first generated text
- [ ] Optimization passes: loop order, cache blocking, SIMD, threads
      (benchmarked individually)
- [ ] Benchmark writeup vs PyTorch CPU
- [ ] Extension: one custom CUDA/Triton kernel (Colab)

## Build & test

```bash
make test
```

Requires a C++17 compiler (clang on macOS works out of the box).

## Layout

- `src/` — the engine (starts with `tensor.h/.cpp`)
- `tests/` — plain-assert test programs, run by `make test`
- `docs/LEARNING.md` — the running lab notebook: what each piece is,
  why it exists, what was measured

## Not covered (on purpose)

Training (this is inference only), GPU support until the core is fast
on CPU, and any model besides GPT-2 small — depth over breadth.
