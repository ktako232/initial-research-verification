#!/usr/bin/env python3

import os
import re
from glob import glob
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# 音声を読み込み、モノラル化して返す
def read_mono(path: str):
    # 音声ファイルを読み込み、モノラル(float32)として返す。
    x, fs = sf.read(path, always_2d=False)
    x = x.astype(np.float32)
    return x, fs

# ピーク正規化（クリップ防止）
def peak_normalize(x: np.ndarray, target: float = 0.98, eps: float = 1e-9):
    # 波形の最大絶対値を target に合わせる。
    peak = float(np.max(np.abs(x)) + eps)
    return x * (target / peak)

# ディレクトリ名から角度を抽出する
def extract_angle_from_name(name: str):
    # 'ang-90', 'ang0', 'ang15' のような文字列から角度(整数)を取得。見つからない場合は None を返す。
    m = re.search(r"ang-?\d+", name)
    if not m:
        return None
    token = m.group(0).replace("ang", "")
    return int(token)

# rirディレクトリ内のファイルを番号順に並べる
def list_rirs_sorted(rir_dir: str, max_n: int = 6):
    # rirディレクトリ内の 'rir-<num>-*.wav' を番号昇順で取得（最大6本）。
    paths = glob(os.path.join(rir_dir, "rir-*.wav"))

    def get_num(p):
        m = re.search(r"rir-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 9999

    paths.sort(key=get_num)
    return paths[:max_n]

# 音声とRIRを畳み込み、正規化して保存
def convolve_and_write(x: np.ndarray, fs: int, rir: np.ndarray, out_path: str):
    # 入力音声 x と RIR を畳み込み、ピーク正規化して保存する。
    y = fftconvolve(x, rir, mode="full")
    y = peak_normalize(y, target=0.98)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, fs)
    print(f"[OK] wrote {out_path}")

# メイン処理（角度ごとに音声×RIRを畳み込み）
def process_all(libri_dir: str, rir_root: str, out_root: str):
    flac_paths = sorted(glob(os.path.join(libri_dir, "*.flac"))) # Libri-Lightの.flacファイル一覧

    # サンプルレート基準取得
    _, fs0 = read_mono(flac_paths[0])

    # 角度フォルダごとに処理
    for name in sorted(os.listdir(rir_root)):
        angle = extract_angle_from_name(name)
        if angle is None: # 角度が含まれないフォルダはスキップ(desktop.ini)
            continue  

        angle_dir = os.path.join(rir_root, name)
        if not os.path.isdir(angle_dir):
            continue

        rir_paths = list_rirs_sorted(angle_dir)
        if len(rir_paths) == 0:
            print(f"[WARN] no rir files in {angle_dir}")
            continue

        # RIR読み込みと正規化
        rirs = []
        for path in rir_paths:
            rir, fs_rir = read_mono(path)
            if fs_rir != fs0:
                print(f"[ERROR] sample rate mismatch: {fs_rir} != {fs0}")
                return
            if np.max(np.abs(rir)) > 0:
                rir = rir / np.max(np.abs(rir))
            rirs.append(rir)

        print(f"[INFO] angle={angle}, RIRs={len(rirs)}")

        # 各音声に対して畳み込み
        for flac_path in flac_paths:
            x, fs_x = read_mono(flac_path)
            if fs_x != fs0:
                print(f"[ERROR] flac sample rate mismatch: {fs_x} != {fs0}")
                return
            if np.max(np.abs(x)) > 0:
                x = peak_normalize(x, 0.95)

            base = Path(flac_path).stem
            for j, rir in enumerate(rirs, start=1):
                out_dir = os.path.join(out_root, f"ang_{angle}", f"Libri{j}")
                out_path = os.path.join(out_dir, f"src_{base}.wav")
                if os.path.exists(out_path):
                    continue  # 再実行時はスキップ
                convolve_and_write(x, fs0, rir, out_path)

    print("[DONE] all processing complete.")

if __name__ == "__main__":
    LIBRI_DIR = "./libri_light_20h_raw"
    RIR_ROOT = "./rir"
    OUT_ROOT = "./sound_src"
    process_all(LIBRI_DIR, RIR_ROOT, OUT_ROOT)

