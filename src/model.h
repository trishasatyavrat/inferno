#pragma once
#include "tensor.h"
#include <vector>

// The GPT-2 model: its shape (config), its parameters (weights), and
// the forward pass that turns token ids into next-token logits.
//
// ops.h holds the individual layers; this file holds the *wiring* -
// the order GPT-2 applies them in and the weights each one reads.
namespace inferno {

struct GPT2Config {
    size_t n_vocab = 50257;  // BPE vocabulary size
    size_t n_ctx   = 1024;   // longest sequence the position table covers
    size_t n_embd  = 768;    // channels per token
    size_t n_head  = 12;
    size_t n_layer = 12;
};

// One transformer block's parameters, named as in the released
// checkpoint (ln_1, attn.c_attn, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj).
struct BlockWeights {
    Tensor ln1_g, ln1_b;       // (C,)
    Tensor w_qkv, b_qkv;       // (C, 3C), (3C,)
    Tensor w_attn_proj, b_attn_proj;  // (C, C), (C,)
    Tensor ln2_g, ln2_b;       // (C,)
    Tensor w_fc, b_fc;         // (C, 4C), (4C,)
    Tensor w_mlp_proj, b_mlp_proj;    // (4C, C), (C,)
};

struct GPT2Weights {
    GPT2Config cfg;
    Tensor wte;   // (n_vocab, C) token embeddings; also the output head (tied)
    Tensor wpe;   // (n_ctx, C)   position embeddings
    std::vector<BlockWeights> blocks;
    Tensor lnf_g, lnf_b;  // final LayerNorm
};

// One block: x + attn(ln1(x)), then x + mlp(ln2(x)).
// "Pre-norm" - LayerNorm goes *before* each sublayer, not after; that is
// the GPT-2 ordering (the original Transformer was post-norm).
Tensor transformer_block(const Tensor& x, const BlockWeights& w, size_t n_head);

// Full forward pass: token ids -> logits (T, n_vocab). Row t is the
// model's score for every possible next token after position t.
Tensor gpt2_forward(const GPT2Weights& model, const std::vector<int>& tokens);

} // namespace inferno
