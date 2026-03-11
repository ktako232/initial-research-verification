#!/usr/bin/env python3

import os
import csv
import json
import argparse
import io
import random
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# =========================
# Global settings
# =========================

ANGLE_CLASSES = list(range(-90, 91, 15))
NUM_ANGLE_CLASSES = len(ANGLE_CLASSES)

# =========================
# Utility functions
# =========================

def angle_to_class(angle_deg):
    angle_deg = float(angle_deg)
    return min(range(len(ANGLE_CLASSES)), key=lambda i: abs(ANGLE_CLASSES[i] - angle_deg))

def class_to_angle(idx):
    return ANGLE_CLASSES[int(idx)]

def parse_angle_from_dirname(name):
    s = name.replace("ang_", "")
    try:
        return float(s)
    except Exception:
        return None

# =========================
# Dataset
# =========================

class PrecomputedMelDataset(Dataset):
    def __init__(self, csv_path):
        self.items = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.items.append((r["npy_path"], int(r["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npy_path, label = self.items[idx]
        mel = np.load(npy_path)                         # (n_mels, T)
        mel = torch.from_numpy(mel).unsqueeze(0).float()  # (1, n_mels, T)
        return mel, label

# =========================
# Model
# =========================

class SpeakerDirectionCNNSimple(nn.Module):
    def __init__(self, n_mels=64, n_angle_classes=NUM_ANGLE_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, n_angle_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# Precompute
# =========================

def precompute(root_dir, out_dir, sample_rate=16000, n_mels=64):
    os.makedirs(out_dir, exist_ok=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=n_mels
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power")

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["wav_path", "npy_path", "label", "angle_deg"])

        for ang in sorted(os.listdir(root_dir)):
            ang_path = os.path.join(root_dir, ang)
            if not os.path.isdir(ang_path):
                continue

            angle_deg = parse_angle_from_dirname(ang)
            if angle_deg is None:
                continue

            label = angle_to_class(angle_deg)

            for root, _, files in os.walk(ang_path):
                rel = os.path.relpath(root, ang_path)
                out_sub = os.path.join(out_dir, ang) if rel == "." else os.path.join(out_dir, ang, rel)
                os.makedirs(out_sub, exist_ok=True)

                for fn in files:
                    if not fn.lower().endswith(".wav"):
                        continue

                    wav_path = os.path.join(root, fn)
                    try:
                        wav, sr = sf.read(wav_path, dtype="float32")
                    except Exception as e:
                        print(f"[SKIP] cannot read: {wav_path} ({e})")
                        continue
                    wav = torch.from_numpy(wav).unsqueeze(0)
                    mel = mel_transform(wav)
                    mel_db = db(mel).squeeze(0).numpy().astype(np.float32)

                    npy_path = os.path.join(out_sub, os.path.splitext(fn)[0] + ".npy")
                    np.save(npy_path, mel_db)

                    writer.writerow([wav_path, npy_path, int(label), float(angle_deg)])

    with open(os.path.join(out_dir, "angle_classes.json"), "w") as jf:
        json.dump({"angle_classes": ANGLE_CLASSES}, jf, indent=2)

    print("precompute finished ->", out_dir)

# =========================
# Visualization utilities
# =========================

def cm_to_image(cm, labels, title=None, normalize=False):
    """
    normalize=False: 件数表示
    normalize=True : 行正規化(=Trueごとに100%になる)の%表示
    """
    cm_disp = cm.astype(np.float64)

    if normalize:
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        # 0除算を避けつつ行正規化して百分率へ
        cm_disp = np.divide(cm_disp, row_sums, out=np.zeros_like(cm_disp), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_disp, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            if normalize:
                text = f"{cm_disp[i, j]:.1f}"   # 例: 12.3 (%)
            else:
                text = str(int(cm[i, j]))       # 件数は元の整数を表示

            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=14,
                color="white" if cm_disp[i, j] > thresh else "black"
            )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def pred_table_to_image(pairs, title):
    fig = plt.figure(figsize=(6, 2 + 0.3 * len(pairs)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)

    table = ax.table(
        cellText=[[t, p] for t, p in pairs],
        colLabels=["true_angle", "pred_angle"],
        loc="center",
        cellLoc="center"
    )
    table.scale(1, 1.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

# =========================
# Checkpoint
# =========================

def save_checkpoint(path, model, optimizer, epoch, best_val_loss, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "history": history
    }, path)

# =========================
# Train
# =========================

def train(precomputed_dir, out_dir, tb_dir, epochs=50, batch_size=32, lr=1e-3, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(precomputed_dir, "train", "metadata.csv")
    if not os.path.exists(train_csv):
        train_csv = os.path.join(precomputed_dir, "metadata.csv")

    val_csv = os.path.join(precomputed_dir, "val", "metadata.csv")
    if not os.path.exists(val_csv):
        val_csv = None

    train_loader = DataLoader(
        PrecomputedMelDataset(train_csv),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = None
    if val_csv:
        val_loader = DataLoader(
            PrecomputedMelDataset(val_csv),
            batch_size=batch_size,
            shuffle=False
        )

    model = SpeakerDirectionCNNSimple().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(tb_dir)
    os.makedirs(out_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    start_epoch = 1

    # --- resume from checkpoint ---
    if resume is not None and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        history = ckpt.get("history", history)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resume from {resume}: start_epoch={start_epoch}")
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    # -----------------------------

    for ep in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running / n
        history["train_loss"].append(train_loss)
        writer.add_scalar("train/loss", train_loss, ep)

        if val_loader:
            model.eval()
            vr = 0.0
            vn = 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_logits = model(vx)
                    v_loss = criterion(v_logits, vy)
                    vr += v_loss.item() * vx.size(0)
                    vn += vx.size(0)
                    preds = v_logits.argmax(dim=1).cpu().numpy()
                    y_pred.extend(preds)
                    y_true.extend(vy.cpu().numpy())

            val_loss = vr / vn
            val_acc = accuracy_score(y_true, y_pred)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            writer.add_scalar("valid/loss", val_loss, ep)
            writer.add_scalar("valid/acc", val_acc, ep)

            cm = confusion_matrix(y_true, y_pred)
            cm_img = cm_to_image(cm, ANGLE_CLASSES, f"Validation epoch {ep}")
            writer.add_image("valid/confusion_matrix", cm_img, ep, dataformats="HWC")

            pairs = [(class_to_angle(t), class_to_angle(p))
                     for t, p in list(zip(y_true, y_pred))[:8]]
            table_img = pred_table_to_image(pairs, f"Validation samples epoch {ep}")
            writer.add_image("valid/sample_predictions", table_img, ep, dataformats="HWC")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(out_dir, "best_model.pth"),
                                model, optimizer, ep, best_val_loss, history)

        save_checkpoint(os.path.join(out_dir, "checkpoint_latest.pth"),
                        model, optimizer, ep, best_val_loss, history)

        print(f"Epoch {ep}: train_loss={train_loss:.4f}")

    writer.close()
    print("training finished")

# =========================
# Evaluate
# =========================

def evaluate(precomputed_dir, model_path, out_dir, tb_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = os.path.join(precomputed_dir, "test", "metadata.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(precomputed_dir, "metadata.csv")

    loader = DataLoader(
        PrecomputedMelDataset(test_csv),
        batch_size=batch_size,
        shuffle=False
    )

    model = SpeakerDirectionCNNSimple().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(out_dir, exist_ok=True)
    
    # =========================
    # Save confusion matrices as PNG
    # =========================
    cm_count_img = cm_to_image(cm, ANGLE_CLASSES, "Test (count)", normalize=False)
    Image.fromarray(cm_count_img).save(os.path.join(out_dir, "confusion_matrix_count.png"))

    cm_percent_img = cm_to_image(cm, ANGLE_CLASSES, "Test (%)", normalize=True)
    Image.fromarray(cm_percent_img).save(os.path.join(out_dir, "confusion_matrix_percent.png"))

    writer = SummaryWriter(tb_dir)
    writer.add_scalar("test/acc", acc, 0)
    # 件数版
    writer.add_image(
        "test/confusion_matrix_count",
        cm_to_image(cm, ANGLE_CLASSES, "Test (count)", normalize=False),
        0,
        dataformats="HWC"
    )

    # %版（行正規化）
    writer.add_image(
        "test/confusion_matrix_percent",
        cm_to_image(cm, ANGLE_CLASSES, "Test (%)", normalize=True),
        0,
        dataformats="HWC"
    )
    writer.close()

    print("test accuracy:", acc)

# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("precompute")
    pre.add_argument("--root", default="./train_dataset")
    pre.add_argument("--out", default="./precomputed/train")

    tr = sub.add_parser("train")
    tr.add_argument("--precomputed", default="./precomputed")
    tr.add_argument("--out", default="./checkpoints")
    tr.add_argument("--tb_dir", default="./runs")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--run_name", default=None)
    tr.add_argument("--resume", default=None)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--precomputed", default="./precomputed")
    ev.add_argument("--model", default="./checkpoints/best_model.pth")
    ev.add_argument("--out", default="./eval")
    ev.add_argument("--tb_dir", default="./runs")
    ev.add_argument("--run_name", default=None)

    args = p.parse_args()

    if args.cmd == "precompute":
        precompute(args.root, args.out)

    elif args.cmd == "train":
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        train(
            args.precomputed,
            os.path.join(args.out, run_name),
            os.path.join(args.tb_dir, run_name),
            args.epochs,
            args.batch_size,
            args.lr,
            resume=args.resume
        )   

    elif args.cmd == "evaluate":
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluate(
            args.precomputed,
            args.model,
            os.path.join(args.out, run_name),
            os.path.join(args.tb_dir, run_name)
        )

if __name__ == "__main__":
    main()
