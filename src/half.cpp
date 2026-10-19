#include "half.h"
#include <stdexcept>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace inferno {

HalfMatrix to_half(const Tensor& t) {
    if (t.shape().size() != 2) throw std::invalid_argument("to_half: 2D tensors only");
    HalfMatrix h;
    h.rows = t.shape()[0];
    h.cols = t.shape()[1];
    h.data.resize(t.size());
    const float* src = t.data();
    size_t i = 0;
#if defined(__ARM_NEON)
    // 8 floats -> 8 halves per step (vcvt rounds to nearest even).
    for (; i + 8 <= t.size(); i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + i));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + i + 4));
        vst1q_f16(reinterpret_cast<float16_t*>(h.data.data() + i), vcombine_f16(lo, hi));
    }
#endif
    for (; i < t.size(); ++i) h.data[i] = static_cast<half>(src[i]);
    return h;
}

Tensor to_float(const HalfMatrix& h) {
    Tensor t({h.rows, h.cols});
    for (size_t i = 0; i < h.size(); ++i) t.data()[i] = static_cast<float>(h.data[i]);
    return t;
}

} // namespace inferno
