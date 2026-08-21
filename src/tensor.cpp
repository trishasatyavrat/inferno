#include "tensor.h"
#include <cassert>
#include <numeric>
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

} // namespace inferno
