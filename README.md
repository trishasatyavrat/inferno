# inferno

A GPT-2 inference engine built C++ (no PyTorch, no ML
libraries.) Loads GPT-2 weights and generates text using tensor
math implemented, then makes it fast: cache blocking, SIMD,
multithreading, with honest benchmarks against PyTorch at every step.

**Why:** every AI framework hides the same core, a few tensor
operations, mostly matrix multiplication, executed as fast as the
hardware allows. This project builds that core in the open, to
understand exactly what runs when a language model generates a word.

## Status

Complete end to end. The full GPT-2 forward pass matches a PyTorch
reference at the real 124M-parameter shape; the released checkpoint
exports into inferno's own weight format; the C++ binary tokenizes a
text prompt (byte-level BPE), generates with a KV cache (greedy /
temperature / top-k, reproducible by seed), and streams text back.
matmul is ~70x its naive baseline (SIMD + threads); the KV cache is
9.7x over full recompute. Honest standing vs PyTorch: 10x slower per
forward pass. Next: thread the single-row matmuls that dominate decode.

- [x] Tensor type (float32, row-major) + naive matmul + tests
- [x] Python bindings (pybind11) + correctness harness vs PyTorch
- [x] Optimization passes: loop order, cache blocking, SIMD
      (each benchmarked and verified against the naive reference)
- [x] Core ops: LayerNorm, softmax, GELU (verified vs PyTorch at
      GPT-2 dimensions, including the 50257-wide vocabulary)
- [x] Multithreading across output rows (rows of C split across cores)
- [x] Causal multi-head attention (fused QKV, 12 heads, verified vs PyTorch)
- [x] MLP block, transformer block, full GPT-2 forward pass (verified vs PyTorch at 124M config)
- [x] Weight file format + loader + exporter from the released checkpoint
- [x] Generation loop: greedy, temperature, top-k sampling, streaming, seeded
- [x] Byte-level BPE tokenizer in C++ (hand-checked merges, exact round trip, tiktoken cross-check)
- [x] KV cache: 9.7x over full recompute (prefill + one row per token)
- [x] End-to-end benchmark vs PyTorch CPU (`make bench-e2e`)
- [ ] Column-split threading for T=1 matmuls (decode is at 25 GB/s; the memory floor is several times higher)
- [ ] Extension: one custom CUDA/Triton kernel (Colab)

## Benchmarks

Square matmul, Apple Silicon, `-O2`, N=512 (`make bench`):

| variant | GFLOP/s | vs naive |
|---|---|---|
| naive triple loop (i-j-k) | 1.8 | 1.0x |
| loop reordered (i-k-j) | 22.4 | 12.4x |
| cache blocked (64x64 tiles) | 19.0 | 10.5x |
| SIMD (NEON) + register blocking | 24.5 | 13.6x |
| SIMD + 11 threads (rows split) | 136.4 | **~70x** |

End to end on the full 124M model (`make bench-e2e`, torch 2.13 CPU):

| | inferno | torch |
|---|---|---|
| forward pass, 64 tokens | 358 ms | 35 ms |
| generate 32 tokens, full recompute | 4.2 tok/s | 39.9 tok/s |
| generate 32 tokens, KV cache | 40.5 tok/s | (torch loop above has no cache) |

Read that honestly: the KV cache is a 9.7x win over our own baseline,
and our cached loop ties PyTorch's *uncached* loop, but PyTorch's
BLAS-backed forward pass is 10x faster than ours. The gap is memory
traffic around the matmuls (per-head copies, allocations) and
single-threaded T=1 matmuls during decode - both are the next targets.

Two results from the matmul work worth stating plainly: cache blocking came in *below* the
plain loop reorder at these sizes (the matrices largely fit in cache
already, so tiling bought overhead rather than locality), and the first
SIMD implementation was slower than no SIMD at all because its inner
loop reloaded and stored C on every iteration - memory traffic, not
arithmetic, was the ceiling. Holding a 1x16 strip of C in NEON registers
across the whole k loop is what actually won. Threading is a clean 4.6x
at N=512 but *loses* at N=64 (thread spawn costs more than the work)
and sags at N=1024 (11 threads contending for bandwidth on the same 4 MB
of B). Details in [docs/LEARNING.md](docs/LEARNING.md).

## Build & test

```bash
make test     # correctness: all matmul variants must agree
make bench    # matmul variants: GFLOP/s each
make bench-e2e # full model vs PyTorch: forward, decode step, generation
make pytest   # harness against PyTorch: every op + the full model (needs .venv)
make weights  # one-time: download GPT-2 small (548 MB) -> weights/gpt2.bin
make inferno  # the CLI: ./build/inferno weights/gpt2.bin --prompt "The capital of France is"
              #   flags: --n --temp --top-k --seed --no-cache --tokenizer
```

Requires a C++17 compiler (clang on macOS works out of the box). The
Python harness needs a venv with torch, numpy and pybind11.

## Layout

- `src/` — the engine: `tensor.h/.cpp` (container + matmul kernels),
  `ops.h/.cpp` (LayerNorm, softmax, GELU, attention, MLP),
  `model.h/.cpp` (GPT-2 wiring), `checkpoint.h/.cpp` (weight file
  format + loader), `main.cpp` (CLI), `bindings.cpp` (Python bridge)
- `tools/export_gpt2.py` — Hugging Face checkpoint -> `weights/gpt2.bin`
  + `weights/tokenizer.bin`
- `tests/` — C++ assert tests + the Python/PyTorch fuzzing harness
- `bench/` — `bench_matmul.cpp` (GFLOP/s per variant),
  `bench_e2e.py` (full model vs PyTorch)
- `docs/LEARNING.md` — the running lab notebook: what each piece is,
  why it exists, what was measured

## Not covered (on purpose)

Training (this is inference only), GPU support until the core is fast
on CPU, and any model besides GPT-2 small — depth over breadth.
