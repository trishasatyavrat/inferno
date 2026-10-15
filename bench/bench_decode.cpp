// Where does a decode step spend its time?
//
// Times each operation of ONE layer of GPT-2 small at decode shape
// (1 new token, L cached positions), plus the final LayerNorm and the
// LM head, and scales the per-layer ops by 12 to give the share of a
// full token. Random weights: speed does not depend on the values.
//
// This is the measurement that should precede every optimization. The
// fused-attention change (Day 14) helped less than hoped because, as
// this table shows, attention was never where the time went.
#include "../src/ops.h"
#include "../src/tensor.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace inferno;
using Clock = std::chrono::steady_clock;

static Tensor rnd(std::vector<size_t> shape, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    Tensor t(shape);
    for (size_t i = 0; i < t.size(); ++i) t.data()[i] = d(g);
    return t;
}

static double time_us(const std::function<void()>& fn, int reps) {
    for (int i = 0; i < 3; ++i) fn();  // warm caches and the thread pool
    auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) fn();
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / reps;
}

int main(int argc, char** argv) {
    const size_t C = 768, H = 12, V = 50257, L = argc > 1 ? std::stoul(argv[1]) : 64;
    const int layers = 12, reps = 200;
    std::mt19937 g(1);
    const float s = 0.02f;

    Tensor x = rnd({1, C}, g, 1.0f);
    Tensor ln_g = rnd({1, C}, g, 0.1f), ln_b = rnd({1, C}, g, 0.1f);
    Tensor w_qkv = rnd({C, 3 * C}, g, s), b_qkv = rnd({1, 3 * C}, g, s);
    Tensor w_proj = rnd({C, C}, g, s), b_proj = rnd({1, C}, g, s);
    Tensor w_fc = rnd({C, 4 * C}, g, s), b_fc = rnd({1, 4 * C}, g, s);
    Tensor w_mp = rnd({4 * C, C}, g, s), b_mp = rnd({1, C}, g, s);
    Tensor wte = rnd({V, C}, g, s);
    Tensor kc = rnd({L + 1, C}, g, 1.0f), vc = rnd({L + 1, C}, g, 1.0f);
    Tensor h_fc = rnd({1, 4 * C}, g, 1.0f);

    struct Row { std::string name; double us; int per_token; };
    std::vector<Row> rows = {
        {"layernorm (x2 per layer)",  time_us([&] { layernorm(x, ln_g, ln_b); }, reps), 2 * layers + 1},
        {"qkv linear  768 -> 2304",   time_us([&] { linear(x, w_qkv, b_qkv); }, reps), layers},
        {"attention core (fused)",    time_us([&] {
             // attention_cached also runs the qkv and proj linears; subtract
             // them below so this row is the score/softmax/value part only.
             Tensor k2 = kc, v2 = vc;
             attention_cached(x, w_qkv, b_qkv, w_proj, b_proj, H, k2, v2, L);
         }, reps), layers},
        {"attn proj   768 -> 768",    time_us([&] { linear(x, w_proj, b_proj); }, reps), layers},
        {"mlp fc      768 -> 3072",   time_us([&] { linear(x, w_fc, b_fc); }, reps), layers},
        {"gelu        3072",          time_us([&] { gelu(h_fc); }, reps), layers},
        {"mlp proj    3072 -> 768",   time_us([&] { linear(h_fc, w_mp, b_mp); }, reps), layers},
        {"residual add (x2 per layer)", time_us([&] { add(x, x); }, reps), 2 * layers},
        {"lm head     768 -> 50257",  time_us([&] { matmul_bt(x, wte); }, 50), 1},
    };
    // attention row: remove the two linears it contains, and the cache copy
    // the benchmark itself makes to keep each rep independent.
    const double copy_us = time_us([&] { Tensor k2 = kc, v2 = vc; (void)k2; (void)v2; }, reps);
    rows[2].us -= rows[1].us + rows[3].us + copy_us;
    if (rows[2].us < 0) rows[2].us = 0;

    double total = 0;
    for (auto& r : rows) total += r.us * r.per_token;
    std::printf("decode step breakdown, GPT-2 small, %zu cached positions\n\n", L);
    std::printf("%-30s %10s %8s %12s %7s\n", "op", "us/call", "calls", "us/token", "share");
    for (auto& r : rows)
        std::printf("%-30s %10.1f %8d %12.0f %6.1f%%\n", r.name.c_str(), r.us, r.per_token,
                    r.us * r.per_token, 100.0 * r.us * r.per_token / total);
    std::printf("%-30s %10s %8s %12.0f\n", "total (sum of parts)", "", "", total);
    return 0;
}
