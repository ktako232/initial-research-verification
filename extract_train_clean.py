#!/usr/bin/env python3

import os
import re
import shutil
from glob import glob

TRAIN_ROOT = "./train_dataset"
CLEAN_ROOT = "./clean_dataset"
OUT_ROOT   = "./sound_src_train_clean"

def mixed_name_to_clean_name(filename: str) -> str | None:
    """
    例:
    src_xxx_c703_n020.wav -> src_xxx_c703.wav
    """
    m = re.match(r"^(.*_c\d+)_n\d+\.wav$", filename)
    if not m:
        return None
    return m.group(1) + ".wav"

def main():
    mixed_wavs = sorted(glob(os.path.join(TRAIN_ROOT, "ang_*", "Libri*", "*.wav")))

    if not mixed_wavs:
        print("[ERROR] train_dataset 配下に wav が見つかりませんでした")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)

    copied = 0
    skipped_existing = 0
    missing = 0
    invalid_name = 0

    for mixed_path in mixed_wavs:
        rel_path = os.path.relpath(mixed_path, TRAIN_ROOT)
        rel_dir = os.path.dirname(rel_path)
        mixed_name = os.path.basename(mixed_path)

        clean_name = mixed_name_to_clean_name(mixed_name)
        if clean_name is None:
            print(f"[WARN] ファイル名形式が想定外です: {mixed_path}")
            invalid_name += 1
            continue

        clean_src_path = os.path.join(CLEAN_ROOT, rel_dir, clean_name)
        out_dir = os.path.join(OUT_ROOT, rel_dir)
        out_path = os.path.join(out_dir, clean_name)

        if not os.path.exists(clean_src_path):
            print(f"[MISSING] {clean_src_path}")
            missing += 1
            continue

        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(out_path):
            skipped_existing += 1
            continue

        shutil.copy2(clean_src_path, out_path)
        copied += 1

    print("\n===== DONE =====")
    print(f"copied           : {copied}")
    print(f"skipped_existing : {skipped_existing}")
    print(f"missing          : {missing}")
    print(f"invalid_name     : {invalid_name}")

if __name__ == "__main__":
    main()
