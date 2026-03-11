#!/usr/bin/env python3
import os
from glob import glob
import numpy as np
import soundfile as sf

def split_into_1sec_chunks(x: np.ndarray, fs: int):
    chunk_len = fs
    chunks = []
    idx = 0
    while idx + chunk_len <= len(x):
        chunks.append(x[idx:idx + chunk_len])
        idx += chunk_len
    return chunks

def main():
    input_root = "./sound_src"
    output_root = "./clean_dataset"
    pattern = "**/*.wav"

    os.makedirs(output_root, exist_ok=True)

    wav_files = sorted(glob(os.path.join(input_root, pattern), recursive=True))
    if not wav_files:
        raise SystemExit(f"[ERROR] no wav files under {input_root}")

    for wav_path in wav_files:
        x, fs = sf.read(wav_path, always_2d=False)
        x = x.astype(np.float32)

        chunks = split_into_1sec_chunks(x, fs)
        if not chunks:
            continue

        rel = os.path.relpath(wav_path, input_root)
        out_path = os.path.join(output_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base = os.path.splitext(out_path)[0]

        for ci, c in enumerate(chunks):
            chunk_path = f"{base}_c{ci:03d}.wav"
            sf.write(chunk_path, c, fs)

if __name__ == "__main__":
    main()
