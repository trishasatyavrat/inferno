#pragma once
#include <cstddef>
#include <functional>

namespace inferno {

// A persistent pool of worker threads, created once per process.
//
// Why: a decode step makes ~60 matmul calls, and until Day 13 each one
// spawned and joined its own threads. Creating a thread costs tens of
// microseconds (stack allocation, kernel scheduling) - comparable to the
// matmul itself at decode sizes. The pool pays that cost once; after
// that, handing out work is an atomic increment and a wake-up.
//
// parallel_for(n, fn) runs fn(0) ... fn(n-1) across the pool, with the
// calling thread doing its share, and returns only when every task has
// finished. Tasks must be independent (ours are: disjoint rows or
// columns of C). Not reentrant, and not for concurrent submitters: one
// thread drives the model, and it must not call parallel_for from
// inside fn.
void parallel_for(size_t n_tasks, const std::function<void(size_t)>& fn);

// Number of threads that share the work: the workers plus the caller.
size_t pool_threads();

} // namespace inferno
