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
#include "model.h"
#include "checkpoint.h"
#include "generate.h"
#include "tokenizer.h"
#include <pybind11/stl.h>

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

static py::array_t<float> py_mlp(Arr x, Arr w_fc, Arr b_fc, Arr w_proj, Arr b_proj) {
    return to_numpy(inferno::mlp(from_numpy(x), from_numpy(w_fc), from_numpy_1d(b_fc),
                                 from_numpy(w_proj), from_numpy_1d(b_proj)));
}

// The whole model, built from a dict of numpy arrays keyed the way the
// released checkpoint names them (wte, wpe, h.0.ln_1.g, ...). This is
// the same structure the on-disk loader will fill in later; here Python
// fills it so the forward pass can be tested against PyTorch at random
// weights before any real checkpoint exists.
struct PyGPT2 {
    inferno::GPT2Weights w;

    PyGPT2(size_t n_vocab, size_t n_ctx, size_t n_embd, size_t n_head, size_t n_layer,
           const py::dict& params) {
        w.cfg = {n_vocab, n_ctx, n_embd, n_head, n_layer};
        auto get = [&](const std::string& key) -> Tensor {
            if (!params.contains(key.c_str()))
                throw std::invalid_argument("missing weight: " + key);
            Arr a = params[key.c_str()].cast<Arr>();
            return a.ndim() == 1 ? from_numpy_1d(a) : from_numpy(a);
        };
        w.wte = get("wte");
        w.wpe = get("wpe");
        for (size_t i = 0; i < n_layer; ++i) {
            const std::string p = "h." + std::to_string(i) + ".";
            inferno::BlockWeights b{
                get(p + "ln_1.g"), get(p + "ln_1.b"),
                get(p + "attn.c_attn.w"), get(p + "attn.c_attn.b"),
                get(p + "attn.c_proj.w"), get(p + "attn.c_proj.b"),
                get(p + "ln_2.g"), get(p + "ln_2.b"),
                get(p + "mlp.c_fc.w"), get(p + "mlp.c_fc.b"),
                get(p + "mlp.c_proj.w"), get(p + "mlp.c_proj.b"),
            };
            w.blocks.push_back(std::move(b));
        }
        w.lnf_g = get("ln_f.g");
        w.lnf_b = get("ln_f.b");
    }

    explicit PyGPT2(const std::string& path) : w(inferno::load_checkpoint(path)) {}

    size_t n_params() const { return inferno::param_count(w); }

    py::array_t<float> forward(const std::vector<int>& tokens) {
        return to_numpy(inferno::gpt2_forward(w, tokens));
    }

    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new,
                              float temperature, size_t top_k, uint32_t seed, bool use_cache) {
        inferno::SampleOptions opt;
        opt.temperature = temperature;
        opt.top_k = top_k;
        opt.seed = seed;
        opt.use_cache = use_cache;
        return inferno::generate(w, prompt, max_new, opt);
    }

    // Feed the chunks through a fresh cache one after another and return
    // the logits of the last chunk. Lets tests check that prefill + steps
    // equals one full forward pass.
    py::array_t<float> forward_chunked(const std::vector<std::vector<int>>& chunks) {
        inferno::KVCache cache(w.cfg);
        Tensor logits;
        for (const auto& c : chunks) logits = inferno::gpt2_forward(w, c, cache);
        return to_numpy(logits);
    }
};

static int py_sample_next(Arr logits, float temperature, size_t top_k, uint32_t seed) {
    auto buf = logits.request();
    inferno::SampleOptions opt;
    opt.temperature = temperature;
    opt.top_k = top_k;
    uint32_t state = inferno::seed_rng(seed);
    return inferno::sample_next(static_cast<const float*>(buf.ptr),
                                static_cast<size_t>(buf.size), opt, state);
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
    m.def("mlp", &py_mlp, "gelu(x @ w_fc + b_fc) @ w_proj + b_proj");

    py::class_<PyGPT2>(m, "GPT2")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, const py::dict&>(),
             py::arg("n_vocab"), py::arg("n_ctx"), py::arg("n_embd"), py::arg("n_head"),
             py::arg("n_layer"), py::arg("params"))
        .def(py::init<const std::string&>(), py::arg("path"),
             "load an INFR checkpoint written by tools/export_gpt2.py")
        .def("n_params", &PyGPT2::n_params)
        .def("forward", &PyGPT2::forward, "token ids -> logits (T, n_vocab)")
        .def("generate", &PyGPT2::generate, "prompt ids -> prompt + generated ids",
             py::arg("prompt"), py::arg("max_new") = 20, py::arg("temperature") = 0.8f,
             py::arg("top_k") = 40, py::arg("seed") = 1, py::arg("use_cache") = true)
        .def("forward_chunked", &PyGPT2::forward_chunked,
             "run chunks through a KV cache in sequence; logits of the last chunk");
    py::class_<inferno::Tokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def("encode", &inferno::Tokenizer::encode, py::arg("text"))
        .def("decode", [](const inferno::Tokenizer& t, const std::vector<int>& ids) {
            // Raw bytes out; Python decodes UTF-8 (a token can split a
            // multi-byte character, so partial output may be invalid).
            return py::bytes(t.decode(ids));
        }, py::arg("ids"))
        .def("vocab_size", &inferno::Tokenizer::vocab_size)
        .def_static("pretokenize", &inferno::Tokenizer::pretokenize, py::arg("text"));
    m.def("sample_next", &py_sample_next, "draw one token id from a logits row",
          py::arg("logits"), py::arg("temperature") = 1.0f, py::arg("top_k") = 0,
          py::arg("seed") = 1);
}
