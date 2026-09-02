// Benchmark: naive vs reordered vs blocked vs SIMD matmul.
//
// Reports GFLOP/s (a matmul does 2*M*N*K floating-point operations).
// Each variant is timed on identical inputs; the naive version is also
// the correctness reference, so speedups here are honest comparisons of
// the same computation, not of different work.
#include "../src/tensor.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace inferno;
using Clock = std::chrono::high_resolution_clock;

static Tensor random_tensor(size_t rows, size_t cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t({rows, cols});
    for (size_t i = 0; i < t.size(); ++i) t.data()[i] = dist(gen);
    return t;
}

static double time_ms(const std::function<Tensor()>& fn, int reps) {
    // One warm-up run so we measure steady state, not cold caches.
    fn();
    auto start = Clock::now();
    for (int r = 0; r < reps; ++r) fn();
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / reps;
}

int main() {
    std::mt19937 gen(42);
    const std::vector<size_t> sizes = {64, 128, 256, 512};

    std::printf("%6s %12s %12s %12s %12s\n",
                "N", "naive", "reordered", "blocked", "simd");
    std::printf("%6s %12s %12s %12s %12s\n",
                "", "GFLOP/s", "GFLOP/s", "GFLOP/s", "GFLOP/s");

    for (size_t n : sizes) {
        Tensor a = random_tensor(n, n, gen);
        Tensor b = random_tensor(n, n, gen);
        const double flops = 2.0 * n * n * n;
        // Fewer reps at large N so the sweep stays quick.
        const int reps = n <= 128 ? 20 : (n <= 256 ? 8 : 3);

        auto gflops = [&](double ms) { return flops / (ms * 1e6); };

        double t_naive = time_ms([&] { return matmul(a, b); }, reps);
        double t_reord = time_ms([&] { return matmul_reordered(a, b); }, reps);
        double t_block = time_ms([&] { return matmul_blocked(a, b); }, reps);
        double t_simd  = time_ms([&] { return matmul_simd(a, b); }, reps);

        std::printf("%6zu %12.2f %12.2f %12.2f %12.2f\n",
                    n, gflops(t_naive), gflops(t_reord),
                    gflops(t_block), gflops(t_simd));
    }

    std::printf("\nSpeedup vs naive at N=512: see the ratio of the last row.\n");
    return 0;
}
