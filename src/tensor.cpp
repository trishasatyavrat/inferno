#include "tensor.h"
#include "pool.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <stdexcept>

namespace inferno {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    size_t total = 1;
    for (size_t d : shape_) total *= d;
    data_.assign(total, 0.0f);
}

float& Tensor::at(size_t i, size_t j) {
    assert(shape_.size() == 2);
    assert(i < shape_[0] && j < shape_[1]);
    // Row-major flattening: row i starts at i * (row length).
    return data_[i * shape_[1] + j];
}

float Tensor::at(size_t i, size_t j) const {
    assert(shape_.size() == 2);
    assert(i < shape_[0] && j < shape_[1]);
    return data_[i * shape_[1] + j];
}

void Tensor::fill(float value) {
    data_.assign(data_.size(), value);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul: 2D tensors only (for now)");
    const size_t M = a.shape()[0], K = a.shape()[1];
    const size_t K2 = b.shape()[0], N = b.shape()[1];
    if (K != K2)
        throw std::invalid_argument("matmul: inner dimensions must match");

    Tensor c({M, N});
    // The famous triple loop. i-j-k order reads B column-wise, which
    // is cache-hostile — measured and fixed in the optimization phase.
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k)
                acc += a.at(i, k) * b.at(k, j);
            c.at(i, j) = acc;
        }
    return c;
}

// --------------------------------------------------------------------
// Optimized variants.
//
// Note on style: these use raw pointers instead of at(). Two reasons -
// at() carries bounds asserts, and the compiler vectorizes flat pointer
// arithmetic far more readily than repeated method calls.
// --------------------------------------------------------------------

static void check_shapes(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul: 2D tensors only");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul: inner dimensions must match");
}

Tensor matmul_reordered(const Tensor& a, const Tensor& b) {
    check_shapes(a, b);
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor c({M, N});
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

    // i-k-j: for a fixed a_ik, stream along row k of B and row i of C.
    // Both are sequential walks, so every cache line fetched is fully
    // consumed. The naive i-j-k order strides down B's columns instead,
    // touching one useful float per cache line at large N.
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            if (a_ik == 0.0f) continue;
            const float* b_row = B + k * N;
            float* c_row = C + i * N;
            for (size_t j = 0; j < N; ++j)
                c_row[j] += a_ik * b_row[j];
        }
    }
    return c;
}

Tensor matmul_blocked(const Tensor& a, const Tensor& b, size_t block) {
    check_shapes(a, b);
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor c({M, N});
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

    // Process block x block tiles so the working set stays cache-resident
    // and is reused many times before eviction. Inner loops keep the
    // i-k-j order from above.
    for (size_t ii = 0; ii < M; ii += block) {
        const size_t i_max = std::min(ii + block, M);
        for (size_t kk = 0; kk < K; kk += block) {
            const size_t k_max = std::min(kk + block, K);
            for (size_t jj = 0; jj < N; jj += block) {
                const size_t j_max = std::min(jj + block, N);
                for (size_t i = ii; i < i_max; ++i) {
                    float* c_row = C + i * N;
                    for (size_t k = kk; k < k_max; ++k) {
                        const float a_ik = A[i * K + k];
                        if (a_ik == 0.0f) continue;
                        const float* b_row = B + k * N;
                        for (size_t j = jj; j < j_max; ++j)
                            c_row[j] += a_ik * b_row[j];
                    }
                }
            }
        }
    }
    return c;
}

// The SIMD kernel over a row range [i0, i1) of C. Pulled out of
// matmul_simd so the threaded version can hand each thread its own
// slice of rows and reuse the exact same inner loop.
static void simd_tile(const float* A, const float* B, float* C,
                      size_t K, size_t N, size_t i0, size_t i1, size_t j0, size_t j1) {
#if defined(__ARM_NEON)
    // Register blocking: hold a 1x16 strip of C in four NEON registers
    // across the ENTIRE k loop, so C is loaded and stored once per strip
    // instead of once per k. The first version of this function did the
    // load/store every iteration and lost to the plain reordered loop -
    // memory traffic, not arithmetic, was the bottleneck.
    for (size_t i = i0; i < i1; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        size_t j = j0;
        for (; j + 16 <= j1; j += 16) {
            float32x4_t c0 = vld1q_f32(c_row + j);
            float32x4_t c1 = vld1q_f32(c_row + j + 4);
            float32x4_t c2 = vld1q_f32(c_row + j + 8);
            float32x4_t c3 = vld1q_f32(c_row + j + 12);
            for (size_t k = 0; k < K; ++k) {
                const float32x4_t va = vdupq_n_f32(a_row[k]);
                const float* b_row = B + k * N + j;
                c0 = vfmaq_f32(c0, va, vld1q_f32(b_row));
                c1 = vfmaq_f32(c1, va, vld1q_f32(b_row + 4));
                c2 = vfmaq_f32(c2, va, vld1q_f32(b_row + 8));
                c3 = vfmaq_f32(c3, va, vld1q_f32(b_row + 12));
            }
            vst1q_f32(c_row + j,      c0);
            vst1q_f32(c_row + j + 4,  c1);
            vst1q_f32(c_row + j + 8,  c2);
            vst1q_f32(c_row + j + 12, c3);
        }
        // Scalar remainder for the tail columns.
        for (; j < j1; ++j) {
            float acc = c_row[j];
            for (size_t k = 0; k < K; ++k)
                acc += a_row[k] * B[k * N + j];
            c_row[j] = acc;
        }
    }
#else
    // Portable fallback: the reordered loop over the same tile.
    for (size_t i = i0; i < i1; ++i) {
        float* c_row = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            const float* b_row = B + k * N;
            for (size_t j = j0; j < j1; ++j) c_row[j] += a_ik * b_row[j];
        }
    }
#endif
}

static void simd_rows(const float* A, const float* B, float* C,
                      size_t K, size_t N, size_t i0, size_t i1) {
    simd_tile(A, B, C, K, N, i0, i1, 0, N);
}

Tensor matmul_simd(const Tensor& a, const Tensor& b) {
    check_shapes(a, b);
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor c({M, N});
    simd_rows(a.data(), b.data(), c.data(), K, N, 0, M);
    return c;
}

Tensor matmul_threaded(const Tensor& a, const Tensor& b, size_t n_threads) {
    check_shapes(a, b);
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor c({M, N});

    if (n_threads == 0) n_threads = pool_threads();
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

    // Threads come from the persistent pool (pool.h), so the per-call
    // cost is a wake-up, not a spawn. The chunking rules below were set
    // when each chunk cost a thread spawn; they still hold - too-small
    // chunks lose to cache effects and wake-up latency - but are now
    // candidates for re-tuning (see LEARNING.md, Day 13).
    // Oversubscribe: hand out ~4 chunks per thread rather than one. The
    // pool's atomic counter then load-balances - a thread that lands on
    // a slow (efficiency) core takes fewer chunks, a fast one takes
    // more, and the job no longer waits for its slowest fixed slice.
    // With one chunk per thread (the spawn-era rule) the N=1024 square
    // case fell from 106 to 34 GFLOP/s the moment workers persisted.
    const size_t oversub = 4;
    const size_t min_rows_per_chunk = 16;
    const size_t row_chunks = std::min(n_threads * oversub, std::max<size_t>(1, M / min_rows_per_chunk));

    if (row_chunks <= 1) {
        // Too few rows to split (M=1 during decode: every weight matrix
        // is read once for a single output row). Split the COLUMNS
        // instead: task t computes C[:, j0:j1], reading only its slice
        // of every row of B. The work is memory-bound - streaming the
        // weights - so the point is to have several cores pulling from
        // memory at once. Strips are multiples of 16 so each task's
        // inner loop stays on the NEON fast path.
        const size_t min_work = 1u << 18;  // ~256K MACs before threads pay off
        const size_t col_chunks = std::min(n_threads * oversub, std::max<size_t>(1, (M * K * N) / min_work));
        if (col_chunks <= 1 || N < 32) {
            simd_rows(A, B, C, K, N, 0, M);
            return c;
        }
        const size_t strip = ((N + col_chunks - 1) / col_chunks + 15) / 16 * 16;
        const size_t n_strips = (N + strip - 1) / strip;
        parallel_for(n_strips, [&](size_t t) {
            const size_t j0 = t * strip;
            simd_tile(A, B, C, K, N, 0, M, j0, std::min(N, j0 + strip));
        });
        return c;
    }

    const size_t rows_per = (M + row_chunks - 1) / row_chunks;
    const size_t n_chunks = (M + rows_per - 1) / rows_per;
    // Each task writes only rows [i0, i1) of C. A and B are read by
    // everyone, which is safe: concurrent reads need no lock.
    parallel_for(n_chunks, [&](size_t t) {
        const size_t i0 = t * rows_per;
        const size_t i1 = std::min(M, i0 + rows_per);
        simd_rows(A, B, C, K, N, i0, i1);
    });
    return c;
}

// Rows [i0, i1) of C = A @ B^T. Each output is a dot product of two
// contiguous K-long rows, accumulated in four NEON lanes and reduced at
// the end - the same "keep the accumulator in registers" idea as
// simd_rows, applied to a reduction instead of a strip of C.
static void bt_rows(const float* A, const float* B, float* C,
                    size_t K, size_t N, size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i) {
        const float* a_row = A + i * K;
        for (size_t j = 0; j < N; ++j) {
            const float* b_row = B + j * K;
            size_t k = 0;
            float acc = 0.0f;
#if defined(__ARM_NEON)
            float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
            for (; k + 8 <= K; k += 8) {
                v0 = vfmaq_f32(v0, vld1q_f32(a_row + k),     vld1q_f32(b_row + k));
                v1 = vfmaq_f32(v1, vld1q_f32(a_row + k + 4), vld1q_f32(b_row + k + 4));
            }
            acc = vaddvq_f32(vaddq_f32(v0, v1));
#endif
            for (; k < K; ++k) acc += a_row[k] * b_row[k];
            C[i * N + j] = acc;
        }
    }
}

Tensor matmul_bt(const Tensor& a, const Tensor& b, size_t n_threads) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul_bt: 2D tensors only");
    if (a.shape()[1] != b.shape()[1])
        throw std::invalid_argument("matmul_bt: inner dimensions must match (A is (M,K), B is (N,K))");
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[0];
    Tensor c({M, N});

    if (n_threads == 0) n_threads = pool_threads();
    // Work per row is N*K; split rows when there is enough of it to be
    // worth a task. The LM head is (T x 768) @ (768 x 50257)^T: at T=1
    // that is a single row of 38M multiply-adds, so for tiny M we split
    // over columns of C (rows of B) instead.
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();
    const size_t oversub = 4;
    if (M >= n_threads * 4) {
        const size_t rows_per = (M + n_threads * oversub - 1) / (n_threads * oversub);
        const size_t n_chunks = (M + rows_per - 1) / rows_per;
        parallel_for(n_chunks, [&](size_t t) {
            const size_t i0 = t * rows_per, i1 = std::min(M, i0 + rows_per);
            bt_rows(A, B, C, K, N, i0, i1);
        });
    } else if (static_cast<double>(M) * N * K >= 1e6 && N >= n_threads) {
        // Column split: task t computes C[:, j0:j1] by treating the
        // corresponding rows of B as its own smaller B.
        const size_t cols_per = (N + n_threads * oversub - 1) / (n_threads * oversub);
        const size_t n_chunks = (N + cols_per - 1) / cols_per;
        parallel_for(n_chunks, [&](size_t t) {
            const size_t j0 = t * cols_per, j1 = std::min(N, j0 + cols_per);
            for (size_t i = 0; i < M; ++i) {
                // Reuse bt_rows on a one-row A against B[j0:j1], writing
                // into a temporary then copying into place.
                std::vector<float> tmp(j1 - j0);
                bt_rows(A + i * K, B + j0 * K, tmp.data(), K, j1 - j0, 0, 1);
                std::copy(tmp.begin(), tmp.end(), C + i * N + j0);
            }
        });
    } else {
        bt_rows(A, B, C, K, N, 0, M);
    }
    return c;
}

} // namespace inferno
