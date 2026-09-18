#include "checkpoint.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace inferno {

namespace {

template <typename T>
T read_pod(std::ifstream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in) throw std::runtime_error("checkpoint: unexpected end of file");
    return v;
}

Tensor read_tensor(std::ifstream& in, std::string& name) {
    const uint32_t name_len = read_pod<uint32_t>(in);
    name.assign(name_len, '\0');
    in.read(name.data(), name_len);
    const uint32_t ndim = read_pod<uint32_t>(in);
    if (ndim == 0 || ndim > 2) throw std::runtime_error("checkpoint: bad ndim for " + name);
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) shape[i] = static_cast<size_t>(read_pod<uint64_t>(in));
    Tensor t(shape);
    // One read straight into the tensor's storage: no per-element loop,
    // no intermediate buffer. 500 MB arrives at disk speed.
    in.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(float)));
    if (!in) throw std::runtime_error("checkpoint: truncated data for " + name);
    return t;
}

Tensor take(std::map<std::string, Tensor>& m, const std::string& key,
            std::vector<size_t> expect) {
    auto it = m.find(key);
    if (it == m.end()) throw std::runtime_error("checkpoint: missing tensor " + key);
    if (it->second.shape() != expect) {
        std::string msg = "checkpoint: " + key + " has shape (";
        for (size_t d : it->second.shape()) msg += std::to_string(d) + ",";
        msg += ") expected (";
        for (size_t d : expect) msg += std::to_string(d) + ",";
        throw std::runtime_error(msg + ")");
    }
    Tensor t = std::move(it->second);
    m.erase(it);
    return t;
}

} // namespace

GPT2Weights load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("checkpoint: cannot open " + path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "INFR", 4) != 0)
        throw std::runtime_error("checkpoint: not an inferno file (bad magic)");
    const uint32_t version = read_pod<uint32_t>(in);
    if (version != 1) throw std::runtime_error("checkpoint: unsupported version " + std::to_string(version));

    GPT2Weights w;
    w.cfg.n_vocab = read_pod<uint32_t>(in);
    w.cfg.n_ctx   = read_pod<uint32_t>(in);
    w.cfg.n_embd  = read_pod<uint32_t>(in);
    w.cfg.n_head  = read_pod<uint32_t>(in);
    w.cfg.n_layer = read_pod<uint32_t>(in);
    const uint32_t count = read_pod<uint32_t>(in);

    std::map<std::string, Tensor> tensors;
    for (uint32_t i = 0; i < count; ++i) {
        std::string name;
        Tensor t = read_tensor(in, name);
        tensors.emplace(std::move(name), std::move(t));
    }

    const size_t C = w.cfg.n_embd;
    w.wte = take(tensors, "wte", {w.cfg.n_vocab, C});
    w.wpe = take(tensors, "wpe", {w.cfg.n_ctx, C});
    for (size_t i = 0; i < w.cfg.n_layer; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        BlockWeights b{
            take(tensors, p + "ln_1.g", {C}), take(tensors, p + "ln_1.b", {C}),
            take(tensors, p + "attn.c_attn.w", {C, 3 * C}), take(tensors, p + "attn.c_attn.b", {3 * C}),
            take(tensors, p + "attn.c_proj.w", {C, C}), take(tensors, p + "attn.c_proj.b", {C}),
            take(tensors, p + "ln_2.g", {C}), take(tensors, p + "ln_2.b", {C}),
            take(tensors, p + "mlp.c_fc.w", {C, 4 * C}), take(tensors, p + "mlp.c_fc.b", {4 * C}),
            take(tensors, p + "mlp.c_proj.w", {4 * C, C}), take(tensors, p + "mlp.c_proj.b", {C}),
        };
        w.blocks.push_back(std::move(b));
    }
    w.lnf_g = take(tensors, "ln_f.g", {C});
    w.lnf_b = take(tensors, "ln_f.b", {C});
    if (!tensors.empty())
        throw std::runtime_error("checkpoint: unexpected extra tensor " + tensors.begin()->first);
    return w;
}

size_t param_count(const GPT2Weights& w) {
    size_t n = w.wte.size() + w.wpe.size() + w.lnf_g.size() + w.lnf_b.size();
    for (const auto& b : w.blocks)
        n += b.ln1_g.size() + b.ln1_b.size() + b.w_qkv.size() + b.b_qkv.size() +
             b.w_attn_proj.size() + b.b_attn_proj.size() + b.ln2_g.size() + b.ln2_b.size() +
             b.w_fc.size() + b.b_fc.size() + b.w_mlp_proj.size() + b.b_mlp_proj.size();
    return n;
}

} // namespace inferno
