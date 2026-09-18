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

// --- Building blocks for the transformer -----------------------------

// Elementwise a + b, same shape. This is the residual connection:
// every block in GPT-2 computes x + f(x), never just f(x).
Tensor add(const Tensor& a, const Tensor& b);

// y = x @ w + b. w is (in, out) and b is (out,) - the layout GPT-2's
// released weights use (HuggingFace calls it Conv1D), chosen so the
// exporter can copy them without transposing. Uses the threaded matmul.
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b);

// Causal multi-head self-attention, one GPT-2 attention layer.
//   x:      (T, C)     T tokens, C channels (768 for GPT-2 small)
//   w_qkv:  (C, 3C)    one fused projection producing Q, K, V
//   b_qkv:  (3C,)
//   w_proj: (C, C)     output projection after the heads are concatenated
//   b_proj: (C,)
//   n_head: heads (12 for GPT-2 small; head size = C / n_head)
// "Causal" means token i may attend to tokens 0..i only - the future is
// masked out - which is what makes the model usable for generation.
Tensor attention(const Tensor& x, const Tensor& w_qkv, const Tensor& b_qkv,
                 const Tensor& w_proj, const Tensor& b_proj, size_t n_head);

// The MLP (feed-forward) half of a transformer block:
//   gelu(x @ w_fc + b_fc) @ w_proj + b_proj
// w_fc is (C, 4C): GPT-2 expands to 3072 channels, applies GELU, and
// projects back to 768. Two-thirds of the model's parameters live here.
Tensor mlp(const Tensor& x, const Tensor& w_fc, const Tensor& b_fc,
           const Tensor& w_proj, const Tensor& b_proj);

} // namespace inferno
