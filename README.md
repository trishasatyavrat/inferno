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

The full GPT-2 forward pass runs and matches a PyTorch reference at the
real 124M-parameter shape, and the released checkpoint can be exported
into inferno's own weight format and loaded by the C++ binary, which generates text from a text prompt
(byte-level BPE tokenizer in C++; greedy / temperature / top-k
sampling, reproducible by seed). matmul is optimized to ~70x its naive
baseline (SIMD + threads). Next: a KV cache, then the end-to-end
benchmark against PyTorch.

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
- [ ] KV cache (measured against the full-recompute baseline)
- [ ] End-to-end benchmark vs PyTorch CPU
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

Two results worth stating plainly: cache blocking came in *below* the
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
make bench    # performance: GFLOP/s per variant
make pytest   # harness against PyTorch: every op + the full model (needs .venv)
make weights  # one-time: download GPT-2 small (548 MB) -> weights/gpt2.bin
make inferno  # the CLI: ./build/inferno weights/gpt2.bin --prompt "The capital of France is"
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
- `bench/` — benchmark harness reporting GFLOP/s per variant
- `docs/LEARNING.md` — the running lab notebook: what each piece is,
  why it exists, what was measured

## Not covered (on purpose)

Training (this is inference only), GPU support until the core is fast
on CPU, and any model besides GPT-2 small — depth over breadth.
