// The inferno command-line tool.
//
//   inferno <weights.bin> [options] <token id> [<token id> ...]
//   inferno <weights.bin> [options] --prompt "text"
//     --tokenizer P  tokenizer file (default: tokenizer.bin next to weights)
//     --n N        tokens to generate (default 20)
//     --temp T     sampling temperature, 0 = greedy (default 0.8)
//     --top-k K    keep the K most likely tokens, 0 = all (default 40)
//     --seed S     random seed (default 1)
//
// Loads a checkpoint, then generates from the prompt and streams the
// output. With --prompt, text goes in and text comes out through the
// tokenizer; with bare ids, ids come out.
#include "checkpoint.h"
#include "tokenizer.h"
#include "generate.h"
#include "model.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cctype>
#include <string>
#include <vector>

using namespace inferno;
using Clock = std::chrono::steady_clock;

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <weights.bin> [--n N] [--temp T] [--top-k K] [--seed S] "
        "[--tokenizer P] (--prompt \"text\" | <token id>...)\n", argv0);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }

    size_t max_new = 20;
    SampleOptions opt;
    opt.temperature = 0.8f;
    opt.top_k = 40;
    opt.seed = 1;
    std::vector<int> prompt;
    std::string prompt_text, tokenizer_path;
    bool have_text = false;
    for (int i = 2; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--n"))          max_new = std::strtoul(need("--n"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--temp"))  opt.temperature = std::strtof(need("--temp"), nullptr);
        else if (!std::strcmp(argv[i], "--top-k")) opt.top_k = std::strtoul(need("--top-k"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--seed"))  opt.seed = static_cast<uint32_t>(std::strtoul(need("--seed"), nullptr, 10));
        else if (!std::strcmp(argv[i], "--prompt")) { prompt_text = need("--prompt"); have_text = true; }
        else if (!std::strcmp(argv[i], "--tokenizer")) tokenizer_path = need("--tokenizer");
        else if (argv[i][0] == '-' && !std::isdigit(static_cast<unsigned char>(argv[i][1]))) { usage(argv[0]); return 2; }
        else prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty() && !have_text) { usage(argv[0]); return 2; }

    // Tokenizer: required for --prompt, optional (for decoding) otherwise.
    if (tokenizer_path.empty()) {
        std::string w = argv[1];
        const size_t slash = w.find_last_of('/');
        tokenizer_path = (slash == std::string::npos ? "" : w.substr(0, slash + 1)) + "tokenizer.bin";
    }
    std::unique_ptr<Tokenizer> tok;
    try {
        tok = std::make_unique<Tokenizer>(tokenizer_path);
    } catch (const std::exception& e) {
        if (have_text) { std::fprintf(stderr, "error: %s (needed for --prompt)\n", e.what()); return 1; }
        // Ids-only mode works without one.
    }
    if (have_text) prompt = tok->encode(prompt_text);

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

    std::printf("prompt (%zu tokens):", prompt.size());
    for (int t : prompt) std::printf(" %d", t);
    std::printf("\n");
    if (tok) {
        // Text mode: echo the prompt, then stream decoded tokens after it.
        std::printf("---\n%s", tok->decode(prompt).c_str());
    } else {
        std::printf("generated:");
    }
    std::fflush(stdout);

    auto t2 = Clock::now();
    size_t produced = 0;
    std::vector<int> out;
    try {
        out = generate(model, prompt, max_new, opt, [&](int id) {
            if (tok) std::fputs(tok->decode({id}).c_str(), stdout);
            else std::printf(" %d", id);
            std::fflush(stdout);  // stream: show each token as it lands
            ++produced;
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    auto t3 = Clock::now();
    const double secs = std::chrono::duration<double>(t3 - t2).count();
    if (tok) std::printf("\n---");
    std::printf("\n%zu tokens in %.2fs (%.1f tok/s, full recompute each step)\n",
                produced, secs, produced / secs);
    return 0;
}
