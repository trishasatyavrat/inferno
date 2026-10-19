#include "half.h"
#include "pool.h"
#include <algorithm>
#include <stdexcept>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace inferno {

HalfMatrix to_half(const Tensor& t) {
    if (t.shape().size() != 2) throw std::invalid_argument("to_half: 2D tensors only");
    HalfMatrix h;
    h.rows = t.shape()[0];
    h.cols = t.shape()[1];
    h.data.resize(t.size());
    const float* src = t.data();
    size_t i = 0;
#if defined(__ARM_NEON)
    // 8 floats -> 8 halves per step (vcvt rounds to nearest even).
    for (; i + 8 <= t.size(); i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + i));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + i + 4));
        vst1q_f16(reinterpret_cast<float16_t*>(h.data.data() + i), vcombine_f16(lo, hi));
    }
#endif
    for (; i < t.size(); ++i) h.data[i] = static_cast<half>(src[i]);
    return h;
}

Tensor to_float(const HalfMatrix& h) {
    Tensor t({h.rows, h.cols});
    for (size_t i = 0; i < h.size(); ++i) t.data()[i] = static_cast<float>(h.data[i]);
    return t;
}

// Rows [i0, i1), columns [j0, j1) of C = A @ B, B in half. The fp32
// kernel's register blocking (a 1x16 strip of C held in four registers
// across the whole k loop), with one change: each row of B arrives as
// two 8-half loads (32 bytes instead of 64) widened to four float32x4.
static void tile_h(const float* A, const half* B, float* C,
                   size_t K, size_t N, size_t i0, size_t i1, size_t j0, size_t j1) {
    for (size_t i = i0; i < i1; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        size_t j = j0;
#if defined(__ARM_NEON)
        for (; j + 16 <= j1; j += 16) {
            float32x4_t c0 = vld1q_f32(c_row + j), c1 = vld1q_f32(c_row + j + 4);
            float32x4_t c2 = vld1q_f32(c_row + j + 8), c3 = vld1q_f32(c_row + j + 12);
            for (size_t k = 0; k < K; ++k) {
                const float32x4_t va = vdupq_n_f32(a_row[k]);
                const float16_t* b = reinterpret_cast<const float16_t*>(B + k * N + j);
                const float16x8_t b0 = vld1q_f16(b), b1 = vld1q_f16(b + 8);
                c0 = vfmaq_f32(c0, va, vcvt_f32_f16(vget_low_f16(b0)));
                c1 = vfmaq_f32(c1, va, vcvt_high_f32_f16(b0));
                c2 = vfmaq_f32(c2, va, vcvt_f32_f16(vget_low_f16(b1)));
                c3 = vfmaq_f32(c3, va, vcvt_high_f32_f16(b1));
            }
            vst1q_f32(c_row + j, c0); vst1q_f32(c_row + j + 4, c1);
            vst1q_f32(c_row + j + 8, c2); vst1q_f32(c_row + j + 12, c3);
        }
#endif
        for (; j < j1; ++j) {
            float acc = c_row[j];
            for (size_t k = 0; k < K; ++k) acc += a_row[k] * static_cast<float>(B[k * N + j]);
            c_row[j] = acc;
        }
    }
}

Tensor matmul_h(const Tensor& a, const HalfMatrix& b) {
    if (a.shape().size() != 2 || a.shape()[1] != b.rows)
        throw std::invalid_argument("matmul_h: A must be (M, K) with K = B.rows");
    const size_t M = a.shape()[0], K = b.rows, N = b.cols;
    Tensor c({M, N});
    const float* A = a.data();
    const half* B = b.data.data();
    float* C = c.data();
    const size_t n_threads = pool_threads(), oversub = 4;

    // Same thresholds as matmul_threaded (Day 12/13), so the fp16 and
    // fp32 paths differ only in the bytes they read.
    const size_t row_chunks = std::min(n_threads * oversub, std::max<size_t>(1, M / 16));
    if (row_chunks <= 1) {
        const size_t col_chunks = std::min(n_threads * oversub, std::max<size_t>(1, (M * K * N) >> 18));
        if (col_chunks <= 1 || N < 32) { tile_h(A, B, C, K, N, 0, M, 0, N); return c; }
        const size_t strip = ((N + col_chunks - 1) / col_chunks + 15) / 16 * 16;
        parallel_for((N + strip - 1) / strip, [&](size_t t) {
            const size_t j0 = t * strip;
            tile_h(A, B, C, K, N, 0, M, j0, std::min(N, j0 + strip));
        });
        return c;
    }
    const size_t rows_per = (M + row_chunks - 1) / row_chunks;
    parallel_for((M + rows_per - 1) / rows_per, [&](size_t t) {
        const size_t i0 = t * rows_per;
        tile_h(A, B, C, K, N, i0, std::min(M, i0 + rows_per), 0, N);
    });
    return c;
}

// Rows [i0, i1), columns [j0, j1) of C = A @ B^T, B (N, K) in half: each
// output is a dot product of a float32 row and a half row, widened 8 at
// a time into two float32x4 accumulators.
static void bt_tile_h(const float* A, const half* B, float* C, size_t K, size_t N,
                      size_t i0, size_t i1, size_t j0, size_t j1) {
    for (size_t i = i0; i < i1; ++i) {
        const float* a_row = A + i * K;
        for (size_t j = j0; j < j1; ++j) {
            const half* b_row = B + j * K;
            size_t k = 0;
            float acc = 0.0f;
#if defined(__ARM_NEON)
            float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
            for (; k + 8 <= K; k += 8) {
                const float16x8_t b = vld1q_f16(reinterpret_cast<const float16_t*>(b_row + k));
                v0 = vfmaq_f32(v0, vld1q_f32(a_row + k), vcvt_f32_f16(vget_low_f16(b)));
                v1 = vfmaq_f32(v1, vld1q_f32(a_row + k + 4), vcvt_high_f32_f16(b));
            }
            acc = vaddvq_f32(vaddq_f32(v0, v1));
#endif
            for (; k < K; ++k) acc += a_row[k] * static_cast<float>(b_row[k]);
            C[i * N + j] = acc;
        }
    }
}

Tensor matmul_bt_h(const Tensor& a, const HalfMatrix& b) {
    if (a.shape().size() != 2 || a.shape()[1] != b.cols)
        throw std::invalid_argument("matmul_bt_h: A must be (M, K) with K = B.cols");
    const size_t M = a.shape()[0], K = b.cols, N = b.rows;
    Tensor c({M, N});
    const float* A = a.data();
    const half* B = b.data.data();
    float* C = c.data();
    const size_t chunks = pool_threads() * 4;
    if (static_cast<double>(M) * N * K < 1e6) { bt_tile_h(A, B, C, K, N, 0, M, 0, N); return c; }
    // Columns of C (rows of B, i.e. vocabulary entries) across the pool:
    // at decode M = 1 and N = 50257, so columns are where the work is.
    const size_t cols_per = (N + chunks - 1) / chunks;
    parallel_for((N + cols_per - 1) / cols_per, [&](size_t t) {
        const size_t j0 = t * cols_per;
        bt_tile_h(A, B, C, K, N, 0, M, j0, std::min(N, j0 + cols_per));
    });
    return c;
}

} // namespace inferno
