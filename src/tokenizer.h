#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// GPT-2's byte-level BPE tokenizer.
//
// Text -> bytes -> "pre-tokens" (words, numbers, punctuation runs, with
// a leading space attached) -> each pre-token merged bottom-up from
// single bytes into vocabulary entries, applying the checkpoint's
// learned merge rules in rank order. Decoding is the trivial inverse:
// concatenate each token's bytes.
//
// File format ("INFT" v1, written by tools/export_gpt2.py):
//   char[4] "INFT", u32 version
//   u32 V, then V x (u16 len, bytes)   - token id -> its raw bytes
//   u32 M, then M x (u32 a, u32 b)     - merge rules in rank order:
//                                        tokens a and b may merge into
//                                        the token whose bytes are a+b
namespace inferno {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return tokens_.size(); }
    const std::string& token_bytes(int id) const { return tokens_.at(static_cast<size_t>(id)); }

    // Split text into the chunks BPE runs on, following GPT-2's regex:
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Exposed for testing. Non-ASCII bytes are treated as letters, which
    // matches the regex for the vast majority of text (see LEARNING.md
    // for the cases where it does not).
    static std::vector<std::string> pretokenize(const std::string& text);

private:
    std::vector<std::string> tokens_;                         // id -> bytes
    std::unordered_map<std::string, int> ids_;                // bytes -> id
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (a,b) -> (rank, merged id)

    static uint64_t pair_key(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }
    std::vector<int> bpe(const std::string& chunk) const;
};

} // namespace inferno
