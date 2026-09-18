// The inferno command-line tool.
//
//   inferno <weights.bin> <token id> [<token id> ...]
//
// Loads a checkpoint, runs the forward pass on the given token ids, and
// prints the model's top-5 predictions for the next token. Token ids
// for now; the tokenizer arrives with the generation loop.
#include "checkpoint.h"
#include "model.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace inferno;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <weights.bin> <token id> [<token id> ...]\n", argv[0]);
        return 2;
    }

    auto t0 = Clock::now();
    GPT2Weights model;
    try {
        model = load_checkpoint(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    auto t1 = Clock::now();
    std::printf("loaded %s: %zu layers, %zu channels, %zu heads, %.1fM parameters in %.2fs\n",
                argv[1], model.cfg.n_layer, model.cfg.n_embd, model.cfg.n_head,
                param_count(model) / 1e6,
                std::chrono::duration<double>(t1 - t0).count());

    std::vector<int> tokens;
    for (int i = 2; i < argc; ++i) tokens.push_back(std::atoi(argv[i]));

    auto t2 = Clock::now();
    Tensor logits = gpt2_forward(model, tokens);
    auto t3 = Clock::now();
    std::printf("forward pass on %zu tokens: %.1f ms\n", tokens.size(),
                std::chrono::duration<double, std::milli>(t3 - t2).count());

    // Top-5 next-token candidates from the last row of logits.
    const size_t T = logits.shape()[0], V = logits.shape()[1];
    const float* last = logits.data() + (T - 1) * V;
    std::vector<size_t> idx(V);
    for (size_t i = 0; i < V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](size_t a, size_t b) { return last[a] > last[b]; });
    std::printf("top-5 next tokens:\n");
    for (size_t r = 0; r < 5; ++r)
        std::printf("  token %6zu  logit %8.3f\n", idx[r], last[idx[r]]);
    return 0;
}
