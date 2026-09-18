#include "tokenizer.h"
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace inferno {

namespace {

template <typename T>
T read_pod(std::ifstream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in) throw std::runtime_error("tokenizer: unexpected end of file");
    return v;
}

bool is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
bool is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;  // non-ASCII: letter
}
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

} // namespace

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("tokenizer: cannot open " + path);
    char magic[4];
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "INFT", 4) != 0)
        throw std::runtime_error("tokenizer: not an inferno tokenizer file");
    if (read_pod<uint32_t>(in) != 1) throw std::runtime_error("tokenizer: unsupported version");

    const uint32_t V = read_pod<uint32_t>(in);
    tokens_.resize(V);
    for (uint32_t i = 0; i < V; ++i) {
        const uint16_t len = read_pod<uint16_t>(in);
        tokens_[i].assign(len, '\0');
        in.read(tokens_[i].data(), len);
        if (!in) throw std::runtime_error("tokenizer: truncated vocab");
        ids_[tokens_[i]] = static_cast<int>(i);
    }
    const uint32_t M = read_pod<uint32_t>(in);
    for (uint32_t r = 0; r < M; ++r) {
        const int a = static_cast<int>(read_pod<uint32_t>(in));
        const int b = static_cast<int>(read_pod<uint32_t>(in));
        if (a < 0 || b < 0 || static_cast<size_t>(a) >= V || static_cast<size_t>(b) >= V)
            throw std::runtime_error("tokenizer: merge references unknown token");
        auto it = ids_.find(tokens_[a] + tokens_[b]);
        if (it == ids_.end())
            throw std::runtime_error("tokenizer: merge result not in vocab (rank " + std::to_string(r) + ")");
        merges_.emplace(pair_key(a, b), std::make_pair(static_cast<int>(r), it->second));
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
    std::vector<std::string> out;
    const size_t n = text.size();
    size_t i = 0;
    auto at = [&](size_t k) -> unsigned char { return static_cast<unsigned char>(text[k]); };

    while (i < n) {
        // Contractions: 's 't 're 've 'm 'll 'd (lowercase only, as in
        // the original regex - a known quirk, "'S" tokenizes differently).
        if (text[i] == '\'' && i + 1 < n) {
            static const char* suf[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            bool matched = false;
            for (const char* s : suf) {
                const size_t L = std::strlen(s);
                if (text.compare(i + 1, L, s) == 0) {
                    out.push_back(text.substr(i, L + 1));
                    i += L + 1;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // Optional single leading space, then a run of one class.
        size_t j = i;
        const bool lead_space = (text[j] == ' ');
        const size_t k = lead_space ? j + 1 : j;
        if (k < n && is_letter(at(k))) {
            j = k;
            while (j < n && is_letter(at(j))) ++j;
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        if (k < n && is_digit(at(k))) {
            j = k;
            while (j < n && is_digit(at(j))) ++j;
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        if (k < n && !is_space(at(k))) {  // punctuation / symbols run
            j = k;
            while (j < n && !is_space(at(j)) && !is_letter(at(j)) && !is_digit(at(j))) ++j;
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // Whitespace run. \s+(?!\S): if non-space follows, all but the
        // last whitespace char form a chunk and the last one attaches
        // to the next word via the " ?" above. At end of text, the whole
        // run is one chunk.
        j = i;
        while (j < n && is_space(at(j))) ++j;
        if (j < n && j - i > 1) {
            out.push_back(text.substr(i, j - i - 1));
            i = j - 1;
            // The remaining single whitespace char: if it is a space it
            // attaches to the next chunk; otherwise (tab, newline) it
            // stands alone.
            if (text[i] != ' ') { out.push_back(text.substr(i, 1)); ++i; }
        } else if (j < n && text[i] != ' ') {
            out.push_back(text.substr(i, 1));
            ++i;
        } else if (j == n) {
            out.push_back(text.substr(i, j - i));
            i = j;
        } else {
            // Single space followed by something that did not take it
            // (cannot happen given the branches above, but never loop).
            out.push_back(text.substr(i, 1));
            ++i;
        }
    }
    return out;
}

std::vector<int> Tokenizer::bpe(const std::string& chunk) const {
    // Start from one token per byte. Every byte is in the vocabulary
    // (that is what makes it "byte-level": no unknown-token case).
    std::vector<int> syms;
    syms.reserve(chunk.size());
    for (unsigned char c : chunk) {
        auto it = ids_.find(std::string(1, static_cast<char>(c)));
        if (it == ids_.end()) throw std::runtime_error("tokenizer: byte not in vocab");
        syms.push_back(it->second);
    }

    // Repeatedly merge the adjacent pair with the lowest rank (= learned
    // earliest = most frequent in the training corpus) until no
    // adjacent pair has a rule. Quadratic in chunk length, but chunks
    // are words, so it does not matter.
    while (syms.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = 0;
        int merged = -1;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merges_.find(pair_key(syms[i], syms[i + 1]));
            if (it != merges_.end() && it->second.first < best_rank) {
                best_rank = it->second.first;
                best_pos = i;
                merged = it->second.second;
            }
        }
        if (merged < 0) break;
        // Replace every occurrence of that pair, left to right, as the
        // reference implementation does.
        std::vector<int> next;
        next.reserve(syms.size());
        for (size_t i = 0; i < syms.size(); ++i) {
            if (i + 1 < syms.size() && syms[i] == syms[best_pos] && syms[i + 1] == syms[best_pos + 1]) {
                next.push_back(merged);
                ++i;
            } else {
                next.push_back(syms[i]);
            }
        }
        syms.swap(next);
    }
    return syms;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    for (const std::string& chunk : pretokenize(text)) {
        std::vector<int> ids = bpe(chunk);
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size())
            throw std::invalid_argument("tokenizer: id out of range");
        out += tokens_[static_cast<size_t>(id)];
    }
    return out;
}

} // namespace inferno
