# Learning Log

Running lab notebook: each day — what we built, what the concepts are,
why they matter, and where to learn more. Written to be understood by
me in an interview, not just executed.

---

## Day 1 (2026-08-21): Tensor + naive matmul

**What we built:** a `Tensor` class (an n-dimensional grid of floats
stored in one flat array) and a deliberately naive matrix multiply,
with hand-checked tests.

**The concepts:**

- **Why a flat array, not nested vectors?** A 2D grid could be
  `vector<vector<float>>`, but each row would live somewhere different
  in memory. One contiguous block means element `(i, j)` is at index
  `i * num_cols + j` ("row-major" layout) — and, critically, that
  walking through it in order matches how the CPU's cache prefetches
  memory. Every speedup later in this project depends on this choice.
  This is EECS 112 cache material, now as a design decision we made.

- **Why is matmul THE operation?** A transformer layer is, almost
  entirely, matrix multiplications (attention scores, projections, the
  MLP). GPT-2 small spends the overwhelming majority of its compute in
  matmul — so making one function fast makes the whole model fast.
  That asymmetry is why GPUs exist.

- **Why write the slow version first?** The naive triple loop is easy
  to verify by hand (see the test's 2x2 worked example). It becomes
  the *oracle*: every clever optimized version must produce the same
  numbers. Fast-but-wrong is worthless; this is the
  correctness-baseline-first discipline.

- **The planted flag:** our loop order (i-j-k) reads matrix B down its
  columns — jumping `N` floats at a time through memory, which defeats
  the cache. We wrote it anyway. When we reorder loops and measure the
  difference in a few days, the speedup will be the architecture
  lecture, live.

**Do now:**
1. `make test` — see it pass.
2. Read `src/tensor.cpp` top to bottom; confirm the `i * shape_[1] + j`
   indexing makes sense by computing where `(1,2)` lands in a 2x3.
3. In the test file, change one expected value and rerun — watch it
   fail, then revert. (Trusting tests means having seen them fail.)

**Resources:**
- "What every programmer should know about memory" (Drepper) — §3-4,
  skim for the cache hierarchy picture
- 3Blue1Brown "Essence of linear algebra" ep. 4 (matrix multiplication
  as composition) — the geometric intuition behind the triple loop
- The GPT-2 paper's architecture section comes later; not yet needed

---

## Day 2 (2026-08-24): Python bindings + the PyTorch oracle

**What we built:** `src/bindings.cpp` — a pybind11 module that lets
Python call our C++ matmul — and `tests/test_vs_torch.py`, which checks
our result against PyTorch's on 8 fixed + 200 random shapes.
`make pytest` builds and runs it.

**The concepts:**

- **Why bindings at all?** PyTorch is itself "Python API, C++ engine."
  Our project mirrors that shape: the engine stays C++, and Python
  becomes the test/experiment cockpit. The `.so` file the build
  produces is a compiled library Python imports like a normal module.

- **The oracle pattern.** We don't hand-check 128x128 matrices; we let
  a battle-tested implementation (PyTorch) be the answer key, and fuzz
  against it with random shapes and a *fixed seed* (failures must be
  reproducible to be debuggable). Every optimization from here on has
  to keep this green — fast-but-wrong is worthless.

- **Why `allclose` and not `==`?** Floating-point addition isn't
  associative: (a+b)+c can differ from a+(b+c) in the last bits.
  Different summation orders → tiny differences → we compare within
  tolerance (rtol/atol), which is standard practice in numerics.

- **Toolchain war stories (real, worth retelling):** (1) The first
  build died with "incompatible architecture (have arm64, need
  x86_64)" — the Anaconda Python was an *Intel* build running under
  Rosetta emulation on this Apple Silicon Mac; native C++ can't load
  into an emulated process. (2) The Homebrew Python replacement
  crashed on a missing libexpat symbol — its bottle expected a newer
  system library than this macOS has. Fix: `uv`-managed Python — a
  self-contained build with zero system dependencies. Lessons:
  architectures must match end-to-end, and pinning/isolating toolchains
  per-project (.venv, uv) is how you make builds reproducible.

**Do now:**
1. `make pytest` — watch the harness pass.
2. In `tests/test_vs_torch.py`, change RTOL to 1e-9 and rerun — see
   float32 reality fail the test, then revert. (Know *why* the
   tolerance exists, don't just accept it.)
3. Read `bindings.cpp`: find where NumPy data is copied into our
   Tensor and back out.

**Resources:**
- pybind11 docs: "First steps" + the NumPy section
- "What Every Computer Scientist Should Know About Floating-Point
  Arithmetic" (Goldberg) — skim §1.5 on rounding; pairs with the
  RTOL experiment

---

## Day 3 (2026-09-02): Making it fast — and being wrong twice first

**What we built:** three optimized matmul variants (`matmul_reordered`,
`matmul_blocked`, `matmul_simd`), a benchmark harness reporting GFLOP/s
(`make bench`), and tests asserting all four implementations agree.

**Measured on this machine (Apple Silicon, -O2), N=512:**

| variant | GFLOP/s | vs naive |
|---|---|---|
| naive (i-j-k) | 1.8 | 1.0x |
| loop reordered (i-k-j) | 22.4 | 12.4x |
| cache blocked (64x64 tiles) | 19.0 | 10.5x |
| SIMD + register blocking | 24.5 | **13.6x** |

**The concepts:**

- **Why the reorder alone gives 12x.** The naive i-j-k loop walks *down*
  a column of B. Memory comes in ~64-byte cache lines, so a column walk
  fetches a whole line and uses 4 bytes of it, then evicts it. Swapping
  to i-k-j walks *along* a row of B instead: every byte of every fetched
  line gets used. Identical arithmetic, ~12x the throughput - the entire
  difference is memory access pattern. This is EECS 112 cache material
  paying off in one number.

- **Blocking LOST to the plain reorder. That was not the plan.** Tiling
  should help by keeping the working set cache-resident - but at N<=512
  the matrices already mostly fit, so tiling added loop overhead and
  interfered with the compiler's own optimization of the simple inner
  loop, for no cache benefit it wasn't already getting. Kept in the repo
  because the result is real: blocking pays off at sizes where the data
  genuinely does not fit, and pretending otherwise would be dishonest.

- **My first SIMD attempt was slower than doing nothing clever.**
  Version one broadcast a_ik and did four FMAs per instruction - and
  came in *below* the plain reordered loop. Diagnosis: the inner loop
  loaded C from memory, did one FMA, and stored C back, every single
  iteration. The bottleneck was never arithmetic, it was memory traffic,
  so adding arithmetic throughput fixed nothing. The rewrite holds a
  1x16 strip of C in four NEON registers across the *entire* k loop -
  one load and one store per strip instead of per k - and that version
  is the fastest of the four. Lesson worth keeping: profile the actual
  bottleneck before optimizing the thing you assume is the bottleneck.

- **Why `-O2` matters to the comparison.** The compiler auto-vectorizes
  the simple reordered loop already, which is why beating it required
  register blocking rather than just "using SIMD." Hand-written
  intrinsics are not automatically faster than a compiler; they win only
  when they express something the compiler cannot infer.

**Do now:**
1. `make bench` on your machine - numbers will differ, the ordering
   should not.
2. Change the block size in `matmul_blocked` (try 16, 32, 128) and
   re-run. The curve has a peak; find roughly where.
3. Read the two SIMD versions in git history (`git log -p src/tensor.cpp`)
   and locate the load/store that made version one slow.

**Resources:**
- Drepper, "What Every Programmer Should Know About Memory" §6.2 (cache
  optimization) - the blocking theory, including when it does not help
- ARM NEON intrinsics reference: `vfmaq_f32`, `vld1q_f32`, `vdupq_n_f32`
- Agner Fog's optimization manuals, ch. on memory access - the general
  form of "memory traffic, not FLOPs, is usually the ceiling"

---

## Day 4 (2026-09-09): The transformer's other three operations

**What we built:** `src/ops.h` / `src/ops.cpp` - LayerNorm, softmax,
and GELU - exposed to Python and verified against PyTorch. With matmul,
these are *every* operation GPT-2 performs. Nothing else is needed to
run the model; the rest is wiring.

**The concepts:**

- **Why a separate ops.cpp.** `tensor.h` owns the container and matmul,
  which change for *performance* reasons. These layers change when the
  *architecture* does. Different reasons to change means different
  files - that is the whole content of "separation of concerns," and
  it is a design question interviewers ask about.

- **LayerNorm, in words.** For each token's feature vector: subtract
  the mean, divide by the standard deviation, then scale and shift by
  learned parameters gamma and beta. It keeps activations in a sane
  range as they flow through 12 stacked blocks. Note it normalizes each
  *row* independently - one token at a time, never across the batch.

- **Softmax's max subtraction is not optional.** Softmax is
  exp(x)/sum(exp(x)), but exp(1000) overflows float32 to infinity and
  the result becomes NaN. Subtracting the row maximum first is
  algebraically a no-op (the factor cancels top and bottom) and
  numerically the difference between working and not. There is a test
  in the harness that feeds it logits of 1000 specifically to prove
  the guard works.

- **GELU: matching the approximation matters.** GPT-2 was trained with
  the *tanh approximation* of GELU, not the exact erf version. Torch's
  default `gelu` is the erf one, so comparing against it made our
  correct code look wrong until the test asked for
  `approximate="tanh"`. Reproducing a model means reproducing its
  arithmetic exactly, approximations included.

- **The bug worth remembering: float32 accumulators drift.** The first
  layernorm passed every shape except 64x50257 - GPT-2's vocabulary
  width - where it diverged from PyTorch by 6.8e-05. Cause: summing
  50,257 float32 values loses low-order bits on every single addition,
  and the error compounds. Fix: accumulate the mean and variance in
  `double` while keeping the data in float32. General rule: **a
  reduction over many elements needs an accumulator wider than the
  elements.** This is exactly why the harness tests real GPT-2
  dimensions instead of only small toy shapes - the bug is invisible
  at 16 columns.

- **Variance formula choice.** We compute the mean of squared
  deviations, not `E[x^2] - E[x]^2`. The second is one pass and looks
  cleverer, but it subtracts two large nearly-equal numbers, which
  catastrophically cancels. Two passes, correct answer.

**Do now:**
1. `make pytest` - all four ops verified.
2. Revert the double accumulators in `layernorm` to float and rerun.
   Watch only the 50257-wide case fail. That is the bug, reproduced.
3. In `check_gelu`, drop `approximate="tanh"` and see how far the erf
   version differs. That gap is why the detail matters.

**Resources:**
- The GPT-2 paper + Karpathy's nanoGPT `model.py` - compare our ops to
  the reference implementation line by line
- "Why does softmax subtract the max?" - any numerical-stability write-up
- Kahan summation (Wikipedia) - the more sophisticated answer to the
  accumulator problem we solved with `double`

---

## Day 5 (2026-09-18): Multithreading - the last CPU lever

**What we built:** `matmul_threaded` - the SIMD kernel with the rows of
C divided across threads (`std::thread`, one per core by default). The
SIMD inner loop was pulled out into `simd_rows(A, B, C, K, N, i0, i1)`
so both `matmul_simd` and the threaded version run *identical* code;
the only difference is who calls it with which row range. The benchmark
gained a threaded column, an N=1024 row, and a block of GPT-2-shaped
cases, because square matrices are a proxy and (T x 768) @ (768 x 2304)
is the real workload.

**Measured (Apple Silicon, 11 hardware threads, -O2):**

| N | naive | reordered | simd | threaded |
|---|---|---|---|---|
| 64 | 2.1 | 26.5 | 43.8 | **9.7** (slower!) |
| 128 | 2.1 | 35.9 | 35.3 | 36.0 |
| 256 | 1.9 | 26.3 | 30.6 | 94.6 |
| 512 | 2.0 | 27.6 | 29.8 | **136.4** |
| 1024 | - | 27.2 | 24.8 | 81.6 |

GPT-2 shapes: at T=1 (generating one token) threading buys nothing
(19.7 vs 21.4 GFLOP/s); at T=64 it is 2.6x; at T=1024 it is 4-5x.

**The concepts:**

- **Why rows, and why no locks.** Row i of C depends on row i of A and
  *all* of B. Two threads never write the same element of C, and
  concurrent *reads* of A and B are always safe. So each thread gets a
  contiguous slice of rows and there is nothing to synchronize until
  `join()`. This is the textbook "embarrassingly parallel" shape; the
  data-race question that makes threading hard does not arise here,
  and it is worth being able to say exactly why.

- **Threading made N=64 slower.** Spawning a thread costs tens of
  microseconds; a 64x64 matmul is a few microseconds of work. The
  function guards against this (at least 16 rows per thread, never
  more threads than that allows) but at N=64 that still means 4
  threads for ~8us of work each. Real libraries keep a persistent
  thread *pool* to avoid paying the spawn cost per call - that is the
  next optimization if this ever matters, and knowing *why* it would
  help is the point.

- **N=1024 dropped from 136 to 82 GFLOP/s.** Every one of the 11
  threads streams through the whole of B (4 MB at N=1024) for each of
  its rows. Eleven threads pulling the same 4 MB through a shared
  cache and memory bus compete for bandwidth; at N=512, B is 1 MB and
  fits comfortably. The fix would be cache blocking *combined with*
  threading, so each thread's working set of B stays small - the tiling
  idea from Day 3 that lost on its own becomes necessary once threads
  multiply the memory traffic. Optimizations interact; the benchmark
  table is how you find out.

- **T=1 is the generation bottleneck and threads do not help it.**
  When the model generates one token at a time, every matmul is
  (1 x 768) @ (768 x N): one row of C, nothing to split. Throughput
  there is bounded by reading the weight matrix from memory once per
  token - memory bandwidth, not compute. This is why inference engines
  obsess over weight size (quantization) and batching, and why a KV
  cache matters: it is the thing that keeps generation from
  recomputing T rows when only one is new.

**Do now:**
1. `make bench` - confirm threaded loses at 64 and wins at 512 on your
   machine.
2. Change `min_rows_per_thread` to 1 and re-run; watch the small-N
   column get worse. Then try 64.
3. Explain out loud why two threads writing to *different rows* of the
   same `std::vector<float>` is safe. (Hint: the vector's memory does
   not move once allocated, and no thread resizes it.)

**Resources:**
- cppreference `std::thread` + `std::thread::hardware_concurrency`
- "Amdahl's law" (any source) - why the T=1 case cannot be helped
- Drepper §6.4 (multi-threaded optimizations) - bandwidth contention
