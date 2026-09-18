// The inferno command-line tool.
//
//   inferno <weights.bin> [options] <token id> [<token id> ...]
//     --n N        tokens to generate (default 20)
//     --temp T     sampling temperature, 0 = greedy (default 0.8)
//     --top-k K    keep the K most likely tokens, 0 = all (default 40)
//     --seed S     random seed (default 1)
//
// Loads a checkpoint, then generates from the prompt token ids and
// prints the new ids as they are produced. Ids in, ids out for now; the
// tokenizer is next.
#include "checkpoint.h"
#include "generate.h"
#include "model.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace inferno;
using Clock = std::chrono::steady_clock;

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <weights.bin> [--n N] [--temp T] [--top-k K] [--seed S] <token id>...\n", argv0);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }

    size_t max_new = 20;
    SampleOptions opt;
    opt.temperature = 0.8f;
    opt.top_k = 40;
    opt.seed = 1;
    std::vector<int> prompt;
    for (int i = 2; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--n"))          max_new = std::strtoul(need("--n"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--temp"))  opt.temperature = std::strtof(need("--temp"), nullptr);
        else if (!std::strcmp(argv[i], "--top-k")) opt.top_k = std::strtoul(need("--top-k"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--seed"))  opt.seed = static_cast<uint32_t>(std::strtoul(need("--seed"), nullptr, 10));
        else if (argv[i][0] == '-' && !std::isdigit(static_cast<unsigned char>(argv[i][1]))) { usage(argv[0]); return 2; }
        else prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty()) { usage(argv[0]); return 2; }

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
                param_count(model) / 1e6, std::chrono::duration<double>(t1 - t0).count());

    std::printf("prompt:");
    for (int t : prompt) std::printf(" %d", t);
    std::printf("\ngenerated:");
    std::fflush(stdout);

    auto t2 = Clock::now();
    size_t produced = 0;
    std::vector<int> out;
    try {
        out = generate(model, prompt, max_new, opt, [&](int tok) {
            std::printf(" %d", tok);
            std::fflush(stdout);  // stream: show each token as it lands
            ++produced;
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    auto t3 = Clock::now();
    const double secs = std::chrono::duration<double>(t3 - t2).count();
    std::printf("\n%zu tokens in %.2fs (%.1f tok/s, full recompute each step)\n",
                produced, secs, produced / secs);
    return 0;
}
