#include "ops.h"
#include <cmath>
#include <stdexcept>
#include <limits>

namespace inferno {

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps) {
    if (x.shape().size() != 2)
        throw std::invalid_argument("layernorm: 2D only");
    const size_t rows = x.shape()[0], cols = x.shape()[1];
    if (gamma.size() != cols || beta.size() != cols)
        throw std::invalid_argument("layernorm: gamma/beta must match last dim");

    Tensor out({rows, cols});
    const float* X = x.data();
    const float* g = gamma.data();
    const float* b = beta.data();
    float* Y = out.data();

    for (size_t i = 0; i < rows; ++i) {
        const float* row = X + i * cols;
        float* orow = Y + i * cols;

        // Accumulate in double, not float. Summing tens of thousands of
        // float32 values loses low-order bits on every add; at GPT-2's
        // vocab width (50257) the drift was large enough to fail the
        // PyTorch comparison. The values stay float32 - only the
        // running total is wider.
        double mean_acc = 0.0;
        for (size_t j = 0; j < cols; ++j) mean_acc += row[j];
        const float mean = static_cast<float>(mean_acc / static_cast<double>(cols));

        // Variance computed as mean of squared deviations rather than
        // E[x^2] - E[x]^2: the latter is one pass but catastrophically
        // cancels when the mean is large relative to the spread.
        double var_acc = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            const double d = static_cast<double>(row[j]) - mean;
            var_acc += d * d;
        }
        const float var = static_cast<float>(var_acc / static_cast<double>(cols));

        const float inv = 1.0f / std::sqrt(var + eps);
        for (size_t j = 0; j < cols; ++j)
            orow[j] = (row[j] - mean) * inv * g[j] + b[j];
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    if (x.shape().size() != 2)
        throw std::invalid_argument("softmax: 2D only");
    const size_t rows = x.shape()[0], cols = x.shape()[1];
    Tensor out({rows, cols});
    const float* X = x.data();
    float* Y = out.data();

    for (size_t i = 0; i < rows; ++i) {
        const float* row = X + i * cols;
        float* orow = Y + i * cols;

        // Subtract the row max before exponentiating. Mathematically a
        // no-op (it cancels in the ratio), numerically essential:
        // exp(1000) overflows float to inf and the result becomes NaN.
        float maxv = row[0];
        for (size_t j = 1; j < cols; ++j) if (row[j] > maxv) maxv = row[j];

        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            orow[j] = std::exp(row[j] - maxv);
            sum += orow[j];
        }
        const float inv = 1.0f / sum;
        for (size_t j = 0; j < cols; ++j) orow[j] *= inv;
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape());
    const float* X = x.data();
    float* Y = out.data();
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = X[i];
        Y[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
    }
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::invalid_argument("add: shapes must match");
    Tensor out(a.shape());
    const float* A = a.data();
    const float* B = b.data();
    float* Y = out.data();
    for (size_t i = 0; i < a.size(); ++i) Y[i] = A[i] + B[i];
    return out;
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
    const size_t out_dim = w.shape()[1];
    if (b.size() != out_dim)
        throw std::invalid_argument("linear: bias must match output dim");
    Tensor y = matmul_threaded(x, w);
    const size_t rows = y.shape()[0];
    const float* B = b.data();
    float* Y = y.data();
    // Broadcast the bias down every row.
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < out_dim; ++j) Y[i * out_dim + j] += B[j];
    return y;
}

// Copy columns [c0, c0+width) of a (rows, cols) tensor into a new
// (rows, width) tensor. Attention slices Q, K, V per head this way.
static Tensor slice_cols(const Tensor& t, size_t c0, size_t width) {
    const size_t rows = t.shape()[0], cols = t.shape()[1];
    Tensor out({rows, width});
    for (size_t i = 0; i < rows; ++i)
        std::copy(t.data() + i * cols + c0, t.data() + i * cols + c0 + width,
                  out.data() + i * width);
    return out;
}

// The transpose of a column slice: (rows, width) -> (width, rows).
// Materialized so K^T can go through the same optimized matmul as
// everything else, rather than needing a second strided kernel.
static Tensor slice_cols_transposed(const Tensor& t, size_t c0, size_t width) {
    const size_t rows = t.shape()[0], cols = t.shape()[1];
    Tensor out({width, rows});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < width; ++j)
            out.data()[j * rows + i] = t.data()[i * cols + c0 + j];
    return out;
}

Tensor attention(const Tensor& x, const Tensor& w_qkv, const Tensor& b_qkv,
                 const Tensor& w_proj, const Tensor& b_proj, size_t n_head) {
    const size_t T = x.shape()[0], C = x.shape()[1];
    if (n_head == 0 || C % n_head != 0)
        throw std::invalid_argument("attention: C must be divisible by n_head");
    if (w_qkv.shape()[0] != C || w_qkv.shape()[1] != 3 * C)
        throw std::invalid_argument("attention: w_qkv must be (C, 3C)");
    const size_t hs = C / n_head;  // head size: 64 for GPT-2 small
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));

    // One matmul produces Q, K and V side by side: columns [0,C) are Q,
    // [C,2C) are K, [2C,3C) are V. Fusing them is a performance choice
    // GPT-2 made (one big matmul beats three), and we inherit it.
    Tensor qkv = linear(x, w_qkv, b_qkv);

    Tensor concat({T, C});  // every head's output, side by side
    for (size_t h = 0; h < n_head; ++h) {
        Tensor q  = slice_cols(qkv, h * hs, hs);                 // (T, hs)
        Tensor kT = slice_cols_transposed(qkv, C + h * hs, hs);  // (hs, T)
        Tensor v  = slice_cols(qkv, 2 * C + h * hs, hs);         // (T, hs)

        // scores[i][j] = how much token i attends to token j.
        Tensor scores = matmul_threaded(q, kT);                  // (T, T)
        float* S = scores.data();
        for (size_t i = 0; i < T; ++i) {
            for (size_t j = 0; j < T; ++j) {
                if (j > i)
                    // Causal mask: -inf becomes exp(-inf) = 0 in softmax,
                    // so token i puts zero weight on any token after it.
                    S[i * T + j] = -std::numeric_limits<float>::infinity();
                else
                    // Scale by 1/sqrt(head size) so dot products of
                    // 64-dim vectors do not saturate the softmax.
                    S[i * T + j] *= scale;
            }
        }
        Tensor weights = softmax(scores);                        // rows sum to 1
        Tensor out_h = matmul_threaded(weights, v);              // (T, hs)

        for (size_t i = 0; i < T; ++i)
            std::copy(out_h.data() + i * hs, out_h.data() + (i + 1) * hs,
                      concat.data() + i * C + h * hs);
    }

    return linear(concat, w_proj, b_proj);
}

} // namespace inferno
