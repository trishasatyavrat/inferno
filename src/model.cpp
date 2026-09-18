#include "model.h"
#include "ops.h"
#include <algorithm>
#include <stdexcept>

namespace inferno {

Tensor transformer_block(const Tensor& x, const BlockWeights& w, size_t n_head) {
    // Residual stream: each sublayer *adds* to x rather than replacing
    // it. That is what lets 12 stacked blocks train at all - gradients
    // and information have a straight path through the additions.
    Tensor h = add(x, attention(layernorm(x, w.ln1_g, w.ln1_b),
                                w.w_qkv, w.b_qkv, w.w_attn_proj, w.b_attn_proj, n_head));
    return add(h, mlp(layernorm(h, w.ln2_g, w.ln2_b),
                      w.w_fc, w.b_fc, w.w_mlp_proj, w.b_mlp_proj));
}

// Embedding for tokens occupying positions [start, start+T):
// x[t] = wte[token_t] + wpe[start+t]. The token row says *what* the
// token is, the position row says *where* it is; attention has no
// notion of order by itself, so position must be added in.
static Tensor embed(const GPT2Weights& m, const std::vector<int>& tokens, size_t start) {
    const GPT2Config& c = m.cfg;
    const size_t T = tokens.size();
    if (T == 0) throw std::invalid_argument("gpt2_forward: empty input");
    if (start + T > c.n_ctx) throw std::invalid_argument("gpt2_forward: sequence longer than n_ctx");
    if (m.blocks.size() != c.n_layer)
        throw std::invalid_argument("gpt2_forward: wrong number of blocks for config");
    Tensor x({T, c.n_embd});
    for (size_t t = 0; t < T; ++t) {
        const int tok = tokens[t];
        if (tok < 0 || static_cast<size_t>(tok) >= c.n_vocab)
            throw std::invalid_argument("gpt2_forward: token id out of range");
        const float* te = m.wte.data() + static_cast<size_t>(tok) * c.n_embd;
        const float* pe = m.wpe.data() + (start + t) * c.n_embd;
        float* row = x.data() + t * c.n_embd;
        for (size_t j = 0; j < c.n_embd; ++j) row[j] = te[j] + pe[j];
    }
    return x;
}

Tensor gpt2_forward(const GPT2Weights& m, const std::vector<int>& tokens) {
    const GPT2Config& c = m.cfg;
    Tensor x = embed(m, tokens, 0);

    for (const BlockWeights& w : m.blocks)
        x = transformer_block(x, w, c.n_head);

    x = layernorm(x, m.lnf_g, m.lnf_b);

    // Language-model head: logits = x @ wte^T. GPT-2 ties the output
    // matrix to the input embedding table - the same 50257 x 768 weights
    // read both ways - which saves 38M parameters and is why the head
    // needs a transposed matmul instead of a second weight tensor.
    return matmul_bt(x, m.wte);
}

Tensor gpt2_forward(const GPT2Weights& m, const std::vector<int>& tokens, KVCache& cache) {
    const GPT2Config& c = m.cfg;
    if (cache.k.size() != c.n_layer)
        throw std::invalid_argument("gpt2_forward: cache built for a different model");
    const size_t start = cache.len;
    Tensor x = embed(m, tokens, start);

    // Same block as transformer_block, but attention reads and extends
    // the cache instead of seeing the whole sequence in x.
    for (size_t l = 0; l < c.n_layer; ++l) {
        const BlockWeights& w = m.blocks[l];
        Tensor h = add(x, attention_cached(layernorm(x, w.ln1_g, w.ln1_b),
                                           w.w_qkv, w.b_qkv, w.w_attn_proj, w.b_attn_proj,
                                           c.n_head, cache.k[l], cache.v[l], start));
        x = add(h, mlp(layernorm(h, w.ln2_g, w.ln2_b),
                       w.w_fc, w.b_fc, w.w_mlp_proj, w.b_mlp_proj));
    }
    cache.len = start + tokens.size();

    x = layernorm(x, m.lnf_g, m.lnf_b);
    return matmul_bt(x, m.wte);
}

} // namespace inferno
