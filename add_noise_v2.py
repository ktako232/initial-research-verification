#!/usr/bin/env python3

import os
import numpy as np
import soundfile as sf
import random
import time
from glob import glob

start_time = time.time()

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def peak_normalize(x: np.ndarray, target: float = 0.98, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + eps)
    return x * (target / peak)

def mix_at_snr(clean: np.ndarray, noise_seg: np.ndarray, snr_db: float, eps: float = 1e-12) -> np.ndarray:
    c_r = max(rms(clean), eps)
    n_r = max(rms(noise_seg), eps)
    target_n_r = c_r / (10.0 ** (snr_db / 20.0))
    scale = target_n_r / n_r
    return clean + noise_seg * scale

def split_into_1sec_chunks(x: np.ndarray, fs: int):
    chunk_len = fs
    chunks = []
    idx = 0
    # print(chunk_len)
    # print(len(x))
    # print("xxxxxxx")
    while idx + chunk_len <= len(x):
        chunks.append(x[idx:idx + chunk_len])
        idx += chunk_len
        # print(idx)
    return chunks

def main():
    clean_root = "./sound_src"
    noise_path = "./noise/20250606_123643.wav"
    out_root   = "./sound_src_with_noise"
    snr_db     = 10.0
    pattern    = "**/*.wav"

    wavfiles_count = 0
    os.makedirs(out_root, exist_ok=True)

    noise, fs_n = sf.read(noise_path, always_2d=False)
    noise = noise.astype(np.float32)
    noise_chunks = split_into_1sec_chunks(noise, fs_n)
    if not noise_chunks:
        raise SystemExit("[ERROR] noise too short to make 1-sec chunks")

    wav_files = sorted(glob(os.path.join(clean_root, pattern), recursive=True))
    if not wav_files:
        raise SystemExit(f"[ERROR] no wav files under {clean_root}")

    for i, wav_path in enumerate(wav_files, 1):
        clean, fs = sf.read(wav_path, always_2d=False)
        clean = clean.astype(np.float32)

        if fs != fs_n:
            raise SystemExit(f"[ERROR] sample rate mismatch: {wav_path}")

        clean_chunks = split_into_1sec_chunks(clean, fs)
        if not clean_chunks:
            continue

        rel = os.path.relpath(wav_path, clean_root)
        out_path = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print("********")
        print(os.path.dirname(out_path))
        print("********")
        base = os.path.splitext(out_path)[0]
        
        for ci, c_chunk in enumerate(clean_chunks):
            nj = random.randrange(len(noise_chunks))
            n_chunk = noise_chunks[nj]
            print("ノイズ番号:", nj)
            mixed = mix_at_snr(c_chunk, n_chunk, snr_db)
            mixed = peak_normalize(mixed, 0.98)
            chunk_path = f"{base}_c{ci:03d}_n{nj:03d}.wav"
            print("書き込み先:", chunk_path)
            sf.write(chunk_path, mixed, fs)
            wavfiles_count = wavfiles_count + 1

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"[INFO] processed {i}/{len(wav_files)}")
        print("wavファイルの現在の個数:", wavfiles_count)
        print(f"経過時間: {elapsed:.2f} 秒")
    
    print(f"[DONE] mixed all 1-sec pairs for {len(wav_files)} clean files.")
    print("全てのwavファイルの個数:", wavfiles_count)
    print(f"[PARAM] SNR={snr_db} dB")

if __name__ == "__main__":
    main()

