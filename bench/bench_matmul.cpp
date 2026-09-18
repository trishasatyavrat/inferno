// Benchmark: naive vs reordered vs blocked vs SIMD vs threaded matmul.
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
#include <thread>
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
    const std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    std::printf("%6s %10s %10s %10s %10s %10s\n",
                "N", "naive", "reordered", "blocked", "simd", "threaded");
    std::printf("%6s %10s %10s %10s %10s %10s\n",
                "", "GFLOP/s", "GFLOP/s", "GFLOP/s", "GFLOP/s", "GFLOP/s");

    for (size_t n : sizes) {
        Tensor a = random_tensor(n, n, gen);
        Tensor b = random_tensor(n, n, gen);
        const double flops = 2.0 * n * n * n;
        // Fewer reps at large N so the sweep stays quick.
        const int reps = n <= 128 ? 20 : (n <= 256 ? 8 : (n <= 512 ? 3 : 1));

        auto gflops = [&](double ms) { return flops / (ms * 1e6); };

        // The naive loop at N=1024 takes several seconds per run; skip
        // it there rather than wait, and print a dash.
        double t_naive = n <= 512 ? time_ms([&] { return matmul(a, b); }, reps) : 0.0;
        double t_reord = time_ms([&] { return matmul_reordered(a, b); }, reps);
        double t_block = time_ms([&] { return matmul_blocked(a, b); }, reps);
        double t_simd  = time_ms([&] { return matmul_simd(a, b); }, reps);
        double t_thr   = time_ms([&] { return matmul_threaded(a, b); }, reps);

        if (t_naive > 0.0)
            std::printf("%6zu %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                        n, gflops(t_naive), gflops(t_reord),
                        gflops(t_block), gflops(t_simd), gflops(t_thr));
        else
            std::printf("%6zu %10s %10.2f %10.2f %10.2f %10.2f\n",
                        n, "-", gflops(t_reord),
                        gflops(t_block), gflops(t_simd), gflops(t_thr));
    }

    // The shapes GPT-2 small actually runs: a T-token sequence (T x 768)
    // against the attention QKV projection (768 x 2304) and the MLP
    // up-projection (768 x 3072). Square benchmarks are a proxy; these
    // are the real workload.
    std::printf("\nGPT-2 shapes (T x 768) @ (768 x N):\n");
    std::printf("%6s %6s %10s %10s\n", "T", "N", "simd", "threaded");
    for (size_t t : {1, 64, 256, 1024}) {
        for (size_t n : {2304, 3072}) {
            Tensor a = random_tensor(t, 768, gen);
            Tensor b = random_tensor(768, n, gen);
            const double flops = 2.0 * t * 768 * n;
            const int reps = t <= 64 ? 20 : 3;
            auto gflops = [&](double ms) { return flops / (ms * 1e6); };
            double t_simd = time_ms([&] { return matmul_simd(a, b); }, reps);
            double t_thr  = time_ms([&] { return matmul_threaded(a, b); }, reps);
            std::printf("%6zu %6zu %10.2f %10.2f\n", t, n, gflops(t_simd), gflops(t_thr));
        }
    }
    std::printf("\nhardware threads: %u\n", std::thread::hardware_concurrency());
    return 0;
}
