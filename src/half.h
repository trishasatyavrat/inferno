#pragma once
#include "tensor.h"
#include <cstddef>
#include <vector>

// Half-precision (float16) weight storage.
//
// Day 15's profile showed decode is memory-bound: the weight matmuls
// read every weight once per token and already run near the memory
// bus's limit. The only remaining lever is reading fewer bytes. Storing
// weights as IEEE float16 halves them (2 bytes instead of 4); the
// kernels convert back to float32 in registers, so all arithmetic and
// every accumulator stays float32. Activations, LayerNorm parameters,
// biases and the KV cache stay float32 too - they are small.
//
// __fp16 is the ARM storage type for IEEE half: 1 sign bit, 5 exponent
// bits, 10 mantissa bits. About 3 significant decimal digits, range
// +-65504. GPT-2's weights are all well inside that range.
#if !defined(__ARM_FP16_FORMAT_IEEE)
#error "inferno's float16 path needs an ARM target with IEEE __fp16 (Apple Silicon)"
#endif

namespace inferno {

using half = __fp16;

// A 2D matrix of halves, row-major, same layout rules as Tensor.
struct HalfMatrix {
    size_t rows = 0, cols = 0;
    std::vector<half> data;

    size_t size() const { return data.size(); }
    size_t bytes() const { return data.size() * sizeof(half); }
    bool empty() const { return data.empty(); }
};

// Round-to-nearest conversion of a 2D float32 Tensor.
HalfMatrix to_half(const Tensor& t);

// Back to float32 (exact: every half is representable as a float).
Tensor to_float(const HalfMatrix& h);

// C = A @ B with A float32 (M, K) and B half (K, N). Same partitioning as
// matmul_threaded: rows of C across the pool when M is large, 16-wide
// column strips when M is small (decode). B is widened to float32 in
// registers, 8 halves at a time; accumulation is float32.
Tensor matmul_h(const Tensor& a, const HalfMatrix& b);

// C = A @ B^T with A float32 (M, K) and B half (N, K). For the tied LM
// head: logits = x @ wte^T with wte stored as half.
Tensor matmul_bt_h(const Tensor& a, const HalfMatrix& b);

} // namespace inferno
