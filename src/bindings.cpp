// Python bindings via pybind11.
//
// This file is the bridge: it compiles into a shared library
// (inferno_core.*.so) that Python can `import` like any module. Python
// then drives our C++ code — which is exactly how PyTorch itself is
// shaped (Python API, C++ engine underneath).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "ops.h"

namespace py = pybind11;
using inferno::Tensor;

// Convert a NumPy 2D array -> our Tensor (copies the data).
static Tensor from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("expected a 2D array");
    Tensor t({static_cast<size_t>(buf.shape[0]), static_cast<size_t>(buf.shape[1])});
    const float* src = static_cast<const float*>(buf.ptr);
    std::copy(src, src + t.size(), t.data());
    return t;
}

// Convert our Tensor -> a NumPy 2D array (copies the data).
static py::array_t<float> to_numpy(const Tensor& t) {
    auto shape = t.shape();
    py::array_t<float> arr({shape[0], shape[1]});
    std::copy(t.data(), t.data() + t.size(), arr.mutable_data());
    return arr;
}

// numpy in, numpy out — matmul runs entirely in our C++.
static py::array_t<float> py_matmul(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    return to_numpy(inferno::matmul(from_numpy(a), from_numpy(b)));
}

static py::array_t<float> py_layernorm(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> gamma,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> beta,
                                       float eps) {
    // gamma/beta arrive 1D; wrap them as (1, C) so from_numpy accepts them.
    auto g = gamma.reshape({static_cast<py::ssize_t>(1), gamma.size()});
    auto b = beta.reshape({static_cast<py::ssize_t>(1), beta.size()});
    return to_numpy(inferno::layernorm(from_numpy(x), from_numpy(g), from_numpy(b), eps));
}

static py::array_t<float> py_softmax(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    return to_numpy(inferno::softmax(from_numpy(x)));
}

static py::array_t<float> py_gelu(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    return to_numpy(inferno::gelu(from_numpy(x)));
}

using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

// 1D numpy vector -> (1, n) Tensor. Biases arrive this way.
static Tensor from_numpy_1d(Arr v) {
    return from_numpy(v.reshape({static_cast<py::ssize_t>(1), v.size()}));
}

static py::array_t<float> py_linear(Arr x, Arr w, Arr b) {
    return to_numpy(inferno::linear(from_numpy(x), from_numpy(w), from_numpy_1d(b)));
}

static py::array_t<float> py_attention(Arr x, Arr w_qkv, Arr b_qkv, Arr w_proj, Arr b_proj,
                                       size_t n_head) {
    return to_numpy(inferno::attention(from_numpy(x), from_numpy(w_qkv), from_numpy_1d(b_qkv),
                                       from_numpy(w_proj), from_numpy_1d(b_proj), n_head));
}

PYBIND11_MODULE(inferno_core, m) {
    m.doc() = "inferno: hand-built tensor ops, exposed to Python";
    m.def("matmul", &py_matmul, "C = A @ B, computed by inferno's C++ engine");
    m.def("layernorm", &py_layernorm, "row-wise LayerNorm",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    m.def("softmax", &py_softmax, "row-wise softmax");
    m.def("gelu", &py_gelu, "GELU (tanh approximation, as in GPT-2)");
    m.def("linear", &py_linear, "x @ w + b (w is (in, out))");
    m.def("attention", &py_attention, "causal multi-head self-attention, GPT-2 layout",
          py::arg("x"), py::arg("w_qkv"), py::arg("b_qkv"), py::arg("w_proj"), py::arg("b_proj"),
          py::arg("n_head"));
}
