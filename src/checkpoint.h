#pragma once
#include "model.h"
#include <string>

// The on-disk weight format, and the loader that reads it.
//
// Format ("INFR", version 1), little-endian:
//   char[4]  magic "INFR"
//   u32      version (1)
//   u32 x5   n_vocab, n_ctx, n_embd, n_head, n_layer
//   u32      tensor count
//   then per tensor:
//     u32        name length, then the name bytes (no terminator)
//     u32        ndim, then u64 x ndim dims
//     float32 x  product(dims) - the data, row-major
//
// Deliberately simple: no compression, no alignment tricks, names in
// the file so a mismatch is an error message rather than a silent
// misload. tools/export_gpt2.py writes it; this reads it.
namespace inferno {

GPT2Weights load_checkpoint(const std::string& path);

// Total parameter count, for reporting (GPT-2 small: ~124M).
size_t param_count(const GPT2Weights& w);

} // namespace inferno
