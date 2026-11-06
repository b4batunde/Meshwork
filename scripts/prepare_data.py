"""Tokenize text with the GPT-2 BPE (tiktoken) into a flat uint16 .bin file.

Speed and memory benchmarks can use synthetic tokens (omit --data). Use a real
dataset when you want loss curves that mean something.

    # ~100M tokens of FineWeb-Edu (streams; no full download)
    python scripts/prepare_data.py --hf HuggingFaceFW/fineweb-edu --hf-config sample-10BT \
        --max-tokens 100_000_000 --out data/fineweb_100M.bin
    # any local text file
    python scripts/prepare_data.py --text-file corpus.txt --out data/corpus.bin
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="Hugging Face dataset name")
    src.add_argument("--text-file")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-tokens", type=int, default=100_000_000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def docs():
        if args.text_file:
            with open(args.text_file, encoding="utf-8") as f:
                yield f.read()
        else:
            from datasets import load_dataset

            ds = load_dataset(args.hf, args.hf_config, split=args.hf_split, streaming=True)
            for row in ds:
                yield row[args.text_field]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    buf = np.memmap(args.out, dtype=np.uint16, mode="w+", shape=(args.max_tokens,))
    n = 0
    for text in docs():
        ids = enc.encode_ordinary(text) + [eot]
        take = min(len(ids), args.max_tokens - n)
        buf[n : n + take] = ids[:take]
        n += take
        if n and n % 5_000_000 < take:
            print(f"{n:,} tokens", flush=True)
        if n >= args.max_tokens:
            break
    buf.flush()
    del buf
    if n < args.max_tokens:  # shrink file to what we actually wrote
        with open(args.out, "r+b") as f:
            f.truncate(n * 2)
    print(f"wrote {n:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
