#pragma once
#include "tensor.h"

// Neural-network operations, kept separate from tensor.h.
//
// tensor.h owns the container and matmul (the linear-algebra
// primitive); this header owns the layers a transformer is built from.
// The split matters because these have different reasons to change:
// matmul changes for performance, these change when the model
// architecture does.
namespace inferno {

// LayerNorm over the last dimension, applied per row.
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// GPT-2 normalizes each token's feature vector independently, which is
// why this is row-wise rather than over the whole tensor.
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps = 1e-5f);

// Row-wise softmax: turns each row into a probability distribution.
// Used twice in GPT-2 - once inside attention over the scores, once at
// the very end over the vocabulary to pick the next token.
Tensor softmax(const Tensor& x);

// GELU activation, tanh approximation - the exact variant GPT-2 uses:
//   0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The approximation is not a shortcut on our part; it is what the
// released weights were trained with, so matching it is a correctness
// requirement, not a performance choice.
Tensor gelu(const Tensor& x);

} // namespace inferno
