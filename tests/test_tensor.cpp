// Plain-assert tests. Run via `make test`.
// Every future optimization must keep these passing bit-for-bit.
#include "../src/tensor.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

using inferno::Tensor;
using inferno::matmul;

static bool close(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    // Construction and indexing.
    Tensor t({2, 3});
    assert(t.size() == 6);
    t.at(1, 2) = 42.0f;
    assert(close(t.at(1, 2), 42.0f));
    assert(close(t.at(0, 0), 0.0f));

    // Known product, worked by hand:
    // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] @ [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    Tensor a({2, 2}), b({2, 2});
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    b.at(0,0)=5; b.at(0,1)=6; b.at(1,0)=7; b.at(1,1)=8;
    Tensor c = matmul(a, b);
    assert(close(c.at(0,0), 19) && close(c.at(0,1), 22));
    assert(close(c.at(1,0), 43) && close(c.at(1,1), 50));

    // Non-square shapes: (1,3) @ (3,2) -> (1,2).
    Tensor p({1, 3}), q({3, 2});
    p.fill(1.0f);
    q.fill(2.0f);
    Tensor r = matmul(p, q);
    assert(r.shape()[0] == 1 && r.shape()[1] == 2);
    assert(close(r.at(0,0), 6.0f)); // 1*2 + 1*2 + 1*2

    // Every optimized variant must agree with the naive reference.
    // This is the invariant that makes the benchmark meaningful: we are
    // comparing four implementations of the SAME computation.
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t shapes[][3] = {{1,1,1}, {3,5,2}, {16,16,16},
                                {64,64,64}, {65,33,17}, {128,96,64}};
    for (auto& s : shapes) {
        Tensor x({s[0], s[1]}), y({s[1], s[2]});
        for (size_t i = 0; i < x.size(); ++i) x.data()[i] = dist(gen);
        for (size_t i = 0; i < y.size(); ++i) y.data()[i] = dist(gen);
        Tensor ref = matmul(x, y);
        Tensor v1 = inferno::matmul_reordered(x, y);
        Tensor v2 = inferno::matmul_blocked(x, y);
        Tensor v3 = inferno::matmul_simd(x, y);
        for (size_t i = 0; i < ref.size(); ++i) {
            // Tolerance, not equality: the variants sum in a different
            // order, and float addition is not associative.
            assert(std::fabs(ref.data()[i] - v1.data()[i]) < 1e-3f);
            assert(std::fabs(ref.data()[i] - v2.data()[i]) < 1e-3f);
            assert(std::fabs(ref.data()[i] - v3.data()[i]) < 1e-3f);
        }
    }

    std::printf("all tensor tests passed (naive + 3 optimized variants agree)\n");
    return 0;
}
