#include "tensor.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <stdexcept>
#include <thread>

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
static void simd_rows(const float* A, const float* B, float* C,
                      size_t K, size_t N, size_t i0, size_t i1) {
#if defined(__ARM_NEON)
    // Register blocking: hold a 1x16 strip of C in four NEON registers
    // across the ENTIRE k loop, so C is loaded and stored once per strip
    // instead of once per k. The first version of this function did the
    // load/store every iteration and lost to the plain reordered loop -
    // memory traffic, not arithmetic, was the bottleneck.
    for (size_t i = i0; i < i1; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        size_t j = 0;
        for (; j + 16 <= N; j += 16) {
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
        for (; j < N; ++j) {
            float acc = c_row[j];
            for (size_t k = 0; k < K; ++k)
                acc += a_row[k] * B[k * N + j];
            c_row[j] = acc;
        }
    }
#else
    // Portable fallback: the reordered loop over the same row range.
    for (size_t i = i0; i < i1; ++i) {
        float* c_row = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            const float* b_row = B + k * N;
            for (size_t j = 0; j < N; ++j) c_row[j] += a_ik * b_row[j];
        }
    }
#endif
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

    if (n_threads == 0) n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;
    // Spawning a thread costs tens of microseconds. For small matrices
    // that overhead exceeds the work, so give each thread a meaningful
    // chunk: at least 16 rows, and never more threads than rows.
    const size_t min_rows_per_thread = 16;
    n_threads = std::min(n_threads, std::max<size_t>(1, M / min_rows_per_thread));
    if (n_threads <= 1) {
        simd_rows(a.data(), b.data(), c.data(), K, N, 0, M);
        return c;
    }

    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();
    const size_t rows_per = (M + n_threads - 1) / n_threads;

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (size_t t = 0; t < n_threads; ++t) {
        const size_t i0 = t * rows_per;
        const size_t i1 = std::min(M, i0 + rows_per);
        if (i0 >= i1) break;
        // Each thread writes only rows [i0, i1) of C. A and B are read
        // by everyone, which is safe: concurrent reads need no lock.
        pool.emplace_back(simd_rows, A, B, C, K, N, i0, i1);
    }
    for (auto& th : pool) th.join();
    return c;
}

} // namespace inferno
