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
