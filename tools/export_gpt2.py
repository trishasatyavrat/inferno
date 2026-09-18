"""Export GPT-2 weights into inferno's checkpoint format.

Usage:
    python tools/export_gpt2.py                 # -> weights/gpt2.bin (124M)
    python tools/export_gpt2.py --out other.bin

Downloads the released GPT-2 small checkpoint (openai-community/gpt2 on
the Hugging Face Hub, ~548 MB safetensors) and rewrites it as the flat
"INFR" format documented in src/checkpoint.h, plus the tokenizer
(vocab.json + merges.txt, ~1.5 MB) as "INFT" (src/tokenizer.h). Needs `huggingface_hub`
and `safetensors` (`uv pip install huggingface_hub safetensors`) - only
for the download; the file writer below has no dependencies beyond
NumPy so tests can round-trip random weights without any download.
"""
import argparse
import os
import struct
import sys

import numpy as np

CONFIG_GPT2_SMALL = {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768,
                     "n_head": 12, "n_layer": 12}


def write_inferno(path, cfg, params):
    """Write cfg + params (dict name -> float32 ndarray) in INFR v1."""
    with open(path, "wb") as f:
        f.write(b"INFR")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<5I", cfg["n_vocab"], cfg["n_ctx"], cfg["n_embd"],
                            cfg["n_head"], cfg["n_layer"]))
        f.write(struct.pack("<I", len(params)))
        for name, arr in params.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}Q", *arr.shape))
            f.write(arr.tobytes(order="C"))


def bytes_to_unicode():
    """GPT-2's byte <-> printable-character table.

    The released vocab.json is text, so each of the 256 byte values is
    shown as a printable Unicode character (space is 'Ġ'). To get real
    bytes back we invert this table. Copied from the reference encoder.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def write_tokenizer(path, tokens, merges):
    """tokens: list[bytes] indexed by id; merges: list[(id_a, id_b)] in rank order."""
    with open(path, "wb") as f:
        f.write(b"INFT")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(tokens)))
        for t in tokens:
            f.write(struct.pack("<H", len(t)))
            f.write(t)
        f.write(struct.pack("<I", len(merges)))
        for a, b in merges:
            f.write(struct.pack("<II", a, b))


def hf_tokenizer_to_inferno(vocab_json, merges_txt):
    import json
    with open(vocab_json, encoding="utf-8") as f:
        vocab = json.load(f)  # printable string -> id
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = bytes(u2b[ch] for ch in s)
    if any(t is None for t in tokens):
        raise ValueError("vocab ids are not contiguous")
    merges = []
    with open(merges_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#version") or not line.strip():
                continue
            a, b = line.rstrip("\n").split(" ")
            merges.append((vocab[a], vocab[b]))
    return tokens, merges


def hf_state_to_params(state, cfg):
    """Rename Hugging Face GPT-2 keys to inferno's and drop what we don't use.

    HF stores Conv1D weights as (in, out) - the same layout inferno uses -
    so this is a rename, not a transpose. `attn.bias` is the causal-mask
    buffer, not a parameter; `lm_head.weight` is tied to wte.
    """
    def g(key):
        for prefix in ("", "transformer."):
            if prefix + key in state:
                return np.asarray(state[prefix + key], dtype=np.float32)
        raise KeyError(key)

    p = {"wte": g("wte.weight"), "wpe": g("wpe.weight"),
         "ln_f.g": g("ln_f.weight"), "ln_f.b": g("ln_f.bias")}
    for i in range(cfg["n_layer"]):
        h = f"h.{i}."
        p[h + "ln_1.g"] = g(h + "ln_1.weight")
        p[h + "ln_1.b"] = g(h + "ln_1.bias")
        p[h + "attn.c_attn.w"] = g(h + "attn.c_attn.weight")
        p[h + "attn.c_attn.b"] = g(h + "attn.c_attn.bias")
        p[h + "attn.c_proj.w"] = g(h + "attn.c_proj.weight")
        p[h + "attn.c_proj.b"] = g(h + "attn.c_proj.bias")
        p[h + "ln_2.g"] = g(h + "ln_2.weight")
        p[h + "ln_2.b"] = g(h + "ln_2.bias")
        p[h + "mlp.c_fc.w"] = g(h + "mlp.c_fc.weight")
        p[h + "mlp.c_fc.b"] = g(h + "mlp.c_fc.bias")
        p[h + "mlp.c_proj.w"] = g(h + "mlp.c_proj.weight")
        p[h + "mlp.c_proj.b"] = g(h + "mlp.c_proj.bias")
    return p


def download_gpt2_small():
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
    except ImportError:
        sys.exit("need huggingface_hub + safetensors: uv pip install huggingface_hub safetensors")
    path = hf_hub_download("openai-community/gpt2", "model.safetensors")
    vocab = hf_hub_download("openai-community/gpt2", "vocab.json")
    merges = hf_hub_download("openai-community/gpt2", "merges.txt")
    return load_file(path), vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("weights", "gpt2.bin"))
    args = ap.parse_args()
    cfg = CONFIG_GPT2_SMALL
    state, vocab_json, merges_txt = download_gpt2_small()
    params = hf_state_to_params(state, cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_inferno(args.out, cfg, params)
    n = sum(v.size for v in params.values())
    print(f"wrote {args.out}: {n / 1e6:.1f}M parameters, "
          f"{os.path.getsize(args.out) / 1e6:.0f} MB")
    tok_out = os.path.join(os.path.dirname(args.out) or ".", "tokenizer.bin")
    tokens, merges = hf_tokenizer_to_inferno(vocab_json, merges_txt)
    write_tokenizer(tok_out, tokens, merges)
    print(f"wrote {tok_out}: {len(tokens)} tokens, {len(merges)} merges")


if __name__ == "__main__":
    main()
