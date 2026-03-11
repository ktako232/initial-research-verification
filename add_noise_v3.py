#!/usr/bin/env python3

import os
import numpy as np
import soundfile as sf
import random
import time
from glob import glob

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

def voice_activity_ratio(x: np.ndarray, fs: int, frame_ms: float = 10.0, rel_thresh: float = 0.5, eps: float = 1e-9) -> float:
    """
    x: 1秒チャンク想定
    - チャンク全体のRMSを基準に、
      各フレームのRMSが (rel_thresh * chunk_rms) 以上なら「アクティブ」とみなす
    - アクティブ率 = アクティブフレーム数 / 全フレーム数
    """
    if len(x) == 0:
        return 0.0

    chunk_rms = rms(x)
    if chunk_rms < eps:
        # チャンク全体がほぼ無音
        return 0.0

    frame_len = int(fs * frame_ms / 1000.0)
    if frame_len <= 0:
        frame_len = 1

    active = 0
    total = 0

    for start in range(0, len(x), frame_len):
        frame = x[start:start + frame_len]
        if len(frame) == 0:
            break
        total += 1
        if rms(frame) >= rel_thresh * chunk_rms:
            active += 1

    if total == 0:
        return 0.0

    return active / total

def check_voice_quality(chunk: np.ndarray, fs: int, min_voice_activity: float = 0.5, min_rms: float = 0.01) -> bool:
    va_ratio = voice_activity_ratio(chunk, fs)
    chunk_rms = rms(chunk)
    return va_ratio >= min_voice_activity and chunk_rms >= min_rms

def main():
    clean_root = "./sound_src"
    noise_path = "./noise/20250606_123643.wav"
    out_root   = "./sound_src_with_noise"
    snr_db     = 12.0
    pattern    = "**/*.wav"

    min_voice_activity = 0.5

    all_wavfiles_count = 0
    all_elapsed = 0
    os.makedirs(out_root, exist_ok=True)

    noise, fs_n = sf.read(noise_path, always_2d=False)
    noise = noise.astype(np.float32)
    noise_chunks = split_into_1sec_chunks(noise, fs_n)
    if not noise_chunks:
        raise SystemExit("[ERROR] noise too short to make 1-sec chunks")

    angle_dirs = sorted(glob(os.path.join(clean_root, "ang_*")))
    if not angle_dirs:
        raise SystemExit("[ERROR] no angle dirs under clean_root")

    for angle_dir in angle_dirs:

        print(f"\n==========================")
        print(f" START angle: {angle_dir}")
        print("==========================\n")

        start_time = time.time()
        wavfiles_count = 0

        wav_files = sorted(glob(os.path.join(angle_dir, pattern), recursive=True))

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
            base = os.path.splitext(out_path)[0]

            # ---- chunk ごとに書き込み ----
            for ci, c_chunk in enumerate(clean_chunks):
                # パワーを考慮したチェック
                if not check_voice_quality(c_chunk, fs, min_voice_activity=0.5, min_rms=0.01):
                    continue

                nj = random.randrange(len(noise_chunks))
                n_chunk = noise_chunks[nj]

                mixed = mix_at_snr(c_chunk, n_chunk, snr_db)
                mixed = peak_normalize(mixed, 0.98)

                chunk_path = f"{base}_c{ci:03d}_n{nj:03d}.wav"
                sf.write(chunk_path, mixed, fs)

                wavfiles_count += 1

        elapsed = time.time() - start_time
        print(f"[DONE] angle {angle_dir}")
        print(f"書き込んだ wavfile 数: {wavfiles_count}")
        print(f"処理時間: {elapsed:.2f} 秒\n")
        all_wavfiles_count += wavfiles_count
        all_elapsed  += elapsed
    
    print(f"[DONE] mixed all 1-sec pairs for {len(wav_files)} clean files.")
    print("全てのwavファイルの個数:", all_wavfiles_count)
    print("合計処理時間:", all_elapsed)
    print(f"[PARAM] SNR={snr_db} dB")

if __name__ == "__main__":
    main()
