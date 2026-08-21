#pragma once
#include <cstddef>
#include <vector>

namespace inferno {

// A Tensor is an n-dimensional grid of floats stored in one flat,
// contiguous array (row-major: the last dimension varies fastest).
// Contiguity is the whole point — it's what makes the cache-friendly
// optimizations later in this project possible.
class Tensor {
public:
    explicit Tensor(std::vector<size_t> shape);

    // Element access for 2D tensors: t.at(row, col).
    float& at(size_t i, size_t j);
    float at(size_t i, size_t j) const;

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Fill helpers used by tests.
    void fill(float value);

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
};

// C = A @ B for 2D tensors. (M,K) @ (K,N) -> (M,N).
// Deliberately naive triple loop — this is the correctness baseline
// every optimized version must match exactly.
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace inferno
