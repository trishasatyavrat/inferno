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

// --- Optimized variants ---------------------------------------------
// All four produce the same result within float tolerance. `matmul`
// above stays the reference implementation the others are checked
// against; each variant below adds exactly one idea so its speedup can
// be measured in isolation.

// 1. Loop reorder (i-k-j). Same arithmetic, different traversal order:
//    B is now walked along rows instead of down columns, so each cache
//    line loaded from B gets fully used before eviction.
Tensor matmul_reordered(const Tensor& a, const Tensor& b);

// 2. Cache blocking (tiling). Work on sub-blocks small enough that the
//    active slices of A, B and C stay resident in L1/L2 while reused.
Tensor matmul_blocked(const Tensor& a, const Tensor& b, size_t block = 64);

// 3. SIMD. Blocking plus explicit vector instructions (ARM NEON on
//    Apple Silicon), computing 4 columns of C per instruction.
Tensor matmul_simd(const Tensor& a, const Tensor& b);

} // namespace inferno
