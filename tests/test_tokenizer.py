"""Tokenizer tests: a hand-built toy vocabulary, the round-trip property,
and (when the real files are present) a cross-check against tiktoken.

Run: make pytest
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import inferno_core
from export_gpt2 import write_tokenizer


def toy_tokenizer(path):
    """All 256 bytes as tokens 0..255, then a few merges built by hand.

    Merges in rank order: 'h'+'e' -> 'he', 'l'+'l' -> 'll', 'he'+'ll' ->
    'hell', 'hell'+'o' -> 'hello', ' '+'w' -> ' w'. Every merge result
    must exist as a token, so those are appended as ids 256..260.
    """
    tokens = [bytes([i]) for i in range(256)]
    tokens += [b"he", b"ll", b"hell", b"hello", b" w"]
    merges = [(ord("h"), ord("e")), (ord("l"), ord("l")), (256, 257), (258, ord("o")),
              (ord(" "), ord("w"))]
    write_tokenizer(path, tokens, merges)
    return tokens


def check_pretokenize():
    P = inferno_core.Tokenizer.pretokenize
    cases = {
        "hello world": ["hello", " world"],
        "Hello, world!": ["Hello", ",", " world", "!"],
        "I'm here, you're not": ["I", "'m", " here", ",", " you", "'re", " not"],
        "abc 123 def": ["abc", " 123", " def"],
        "two  spaces": ["two", " ", " spaces"],       # \s+(?!\S): one stays, one attaches
        "trailing   ": ["trailing", "   "],           # at end: whole run
        "line\nbreak": ["line", "\n", "break"],
        "tab\tthen": ["tab", "\t", "then"],
        "café au lait": ["café", " au", " lait"],     # non-ASCII bytes treated as letters
        "": [],
        "   ": ["   "],
        "x": ["x"],
        "$100": ["$", "100"],
    }
    for text, want in cases.items():
        got = P(text)
        if got != want:
            raise AssertionError(f"pretokenize({text!r}) = {got!r}, want {want!r}")
        if "".join(got) != text:
            raise AssertionError(f"pretokenize lost characters on {text!r}")


def check_toy():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "toy.bin")
        tokens = toy_tokenizer(path)
        tok = inferno_core.Tokenizer(path)
        if tok.vocab_size() != len(tokens):
            raise AssertionError("vocab size")

        # Worked by hand: h e l l o -> (rank0) he l l o -> (rank1) he ll o
        # -> (rank2) hell o -> (rank3) hello.
        if tok.encode("hello") != [259]:
            raise AssertionError(f"'hello' -> {tok.encode('hello')}, want [259]")
        # "hell" stops at rank 2; "help": he + l + p (no rule for he+l).
        if tok.encode("hell") != [258] or tok.encode("help") != [256, ord("l"), ord("p")]:
            raise AssertionError("partial merges wrong")
        # Two chunks: "hello" and " world" -> [hello] + [' w', o, r, l, d].
        if tok.encode("hello world") != [259, 260, ord("o"), ord("r"), ord("l"), ord("d")]:
            raise AssertionError(f"'hello world' -> {tok.encode('hello world')}")
        # Merges never cross chunk boundaries: "hel lo" has no 'hello'.
        if 259 in tok.encode("hel lo"):
            raise AssertionError("merge crossed a pre-token boundary")

        # Round trip is exact for ANY bytes, including multi-byte UTF-8
        # and characters the vocab has no merges for - the whole point
        # of byte-level BPE.
        samples = ["hello world", "Hello, World! 123", "  leading and trailing  ",
                   "naïve café — “quotes” 日本語 🚀", "tabs\tand\nnewlines\r\n", "", "a"]
        for s in samples:
            ids = tok.encode(s)
            back = tok.decode(ids).decode("utf-8")
            if back != s:
                raise AssertionError(f"round trip failed: {s!r} -> {ids} -> {back!r}")


def check_against_tiktoken():
    """Only runs when weights/tokenizer.bin exists and tiktoken is installed."""
    path = os.path.join(os.path.dirname(__file__), "..", "weights", "tokenizer.bin")
    if not os.path.exists(path):
        return "skipped (no weights/tokenizer.bin - run `make weights`)"
    try:
        import tiktoken
    except ImportError:
        return "skipped (tiktoken not installed: uv pip install tiktoken)"
    enc = tiktoken.get_encoding("gpt2")
    tok = inferno_core.Tokenizer(path)
    texts = ["The capital of France is", "Hello, world! I'm here.", "  two  spaces and\ttabs\n",
             "numbers 12345 and $9.99", "unicode: café naïve 日本語 🚀", "don't won't can't I'll",
             "A long sentence, with commas, semicolons; and (parentheses) - plus dashes."]
    for t in texts:
        ours, theirs = tok.encode(t), enc.encode(t)
        if ours != theirs:
            raise AssertionError(f"tiktoken mismatch on {t!r}:\n ours   {ours}\n theirs {theirs}")
    return f"matches tiktoken on {len(texts)} texts"


def main():
    check_pretokenize()
    check_toy()
    print("tokenizer: pretokenizer cases, hand-worked toy merges, exact round trip")
    print(f"tokenizer vs tiktoken: {check_against_tiktoken()}")


if __name__ == "__main__":
    main()
