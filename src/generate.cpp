#include "generate.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace inferno {

uint32_t seed_rng(uint32_t seed) {
    uint32_t h = seed;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h ? h : 0x9E3779B9u;  // xorshift must never start at zero
}

int sample_next(const float* logits, size_t V, const SampleOptions& opt, uint32_t& rng_state) {
    if (V == 0) throw std::invalid_argument("sample_next: empty logits");

    // Greedy: the single highest-scoring token, no randomness. Also the
    // limit of temperature -> 0, which is why temperature 0 selects it.
    if (opt.temperature <= 0.0f)
        return static_cast<int>(std::max_element(logits, logits + V) - logits);

    // Top-k: sort only the k best (partial_sort is O(V log k), not
    // O(V log V)) and ignore the rest. Cutting the long tail of
    // near-zero-probability tokens is what keeps sampled text coherent;
    // without it the model occasionally picks nonsense that had 0.01%.
    const size_t k = (opt.top_k == 0 || opt.top_k > V) ? V : opt.top_k;
    std::vector<size_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    // Softmax over the survivors at the requested temperature. Dividing
    // logits by T < 1 sharpens the distribution (more greedy); T > 1
    // flattens it (more random). Max-subtracted, in double, as always.
    std::vector<double> probs(k);
    const double maxv = logits[idx[0]];
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) {
        probs[i] = std::exp((logits[idx[i]] - maxv) / opt.temperature);
        sum += probs[i];
    }

    // Draw a uniform number and walk the cumulative distribution. A
    // hand-rolled generator (xorshift) so results are identical across
    // platforms and compilers - std::mt19937 is portable but the
    // distributions on top of it are not.
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    const double u = (rng_state / 4294967296.0) * sum;
    double acc = 0.0;
    for (size_t i = 0; i < k; ++i) {
        acc += probs[i];
        if (u < acc) return static_cast<int>(idx[i]);
    }
    return static_cast<int>(idx[k - 1]);  // rounding fell off the end
}

std::vector<int> generate(const GPT2Weights& model, std::vector<int> tokens,
                          size_t max_new, const SampleOptions& opt,
                          const std::function<void(int)>& on_token) {
    if (tokens.empty()) throw std::invalid_argument("generate: empty prompt");
    uint32_t rng_state = seed_rng(opt.seed);
    const size_t V = model.cfg.n_vocab;

    for (size_t step = 0; step < max_new; ++step) {
        if (tokens.size() >= model.cfg.n_ctx) break;  // the position table ends here
        Tensor logits = gpt2_forward(model, tokens);
        // Only the last row matters: it is the prediction for what comes
        // *after* the final token. Every other row was recomputed for
        // nothing - the waste the KV cache exists to remove.
        const float* last = logits.data() + (logits.shape()[0] - 1) * V;
        const int next = sample_next(last, V, opt, rng_state);
        tokens.push_back(next);
        if (on_token) on_token(next);
    }
    return tokens;
}

} // namespace inferno
