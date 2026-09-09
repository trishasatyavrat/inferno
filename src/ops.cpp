#include "ops.h"
#include <cmath>
#include <stdexcept>

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

} // namespace inferno
