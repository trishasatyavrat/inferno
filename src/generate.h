#pragma once
#include "model.h"
#include <cstdint>
#include <functional>
#include <vector>

// Turning logits into text, one token at a time.
//
// The forward pass gives a score for every vocabulary entry; sampling
// decides which one to emit. Generation is the loop: append the chosen
// token, run the model again, repeat.
namespace inferno {

struct SampleOptions {
    float temperature = 1.0f;  // 0 = greedy (always the argmax)
    size_t top_k = 40;         // 0 = consider the whole vocabulary
    uint32_t seed = 0;         // fixed seed = reproducible output
};

// Turn a user-facing seed into an RNG state. xorshift's first outputs
// from a small seed (1, 2, 3...) are themselves small - the bits have
// not had time to spread - so seed 1 would draw "u = 0.00006" and pick
// the top token every time. Hashing the seed first (murmur3's finalizer)
// scatters it across all 32 bits. Found by the sampling test, not by
// reading the code.
uint32_t seed_rng(uint32_t seed);

// Pick one token from a row of V logits. Applies temperature, keeps
// the top_k highest, softmaxes those, and draws from the result.
int sample_next(const float* logits, size_t V, const SampleOptions& opt, uint32_t& rng_state);

// Autoregressive generation. Returns prompt + up to max_new tokens
// (fewer if the context window n_ctx fills up). on_token, if given, is
// called with each new token as it is produced - that is what lets a
// UI stream output instead of waiting for the whole thing.
//
// This version recomputes the full forward pass over every token at
// every step: O(T^2) work for T tokens. That is the honest baseline the
// KV cache will be measured against.
std::vector<int> generate(const GPT2Weights& model, std::vector<int> tokens,
                          size_t max_new, const SampleOptions& opt,
                          const std::function<void(int)>& on_token = nullptr);

} // namespace inferno
