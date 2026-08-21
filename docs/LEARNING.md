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
