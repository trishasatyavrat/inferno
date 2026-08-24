// Python bindings via pybind11.
//
// This file is the bridge: it compiles into a shared library
// (inferno_core.*.so) that Python can `import` like any module. Python
// then drives our C++ code — which is exactly how PyTorch itself is
// shaped (Python API, C++ engine underneath).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor.h"

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

PYBIND11_MODULE(inferno_core, m) {
    m.doc() = "inferno: hand-built tensor ops, exposed to Python";
    m.def("matmul", &py_matmul, "C = A @ B, computed by inferno's C++ engine");
}
