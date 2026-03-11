#!/usr/bin/env python3

import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

class DirectoryAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, n_mels=64, extensions=(".wav",)):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.data = []  # list of (abs_path, label)
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not classes:
            raise RuntimeError(f"No class dirs under {root_dir}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            cdir = os.path.join(root_dir, c)
            for root, _, files in os.walk(cdir):
                for fn in files:
                    if fn.lower().endswith(extensions):
                        abs_path = os.path.join(root, fn)
                        self.data.append((abs_path, self.class_to_idx[c]))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=n_mels,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        target_len = self.sample_rate
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        elif wav.shape[1] < target_len:
            pad = target_len - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        mel = self.mel(wav)           # (1, n_mels, T')
        mel_db = self.db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db, label

class ConvNet(nn.Module):
    def __init__(self, n_mels=64, n_classes=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    train_root = "./train_dataset"
    val_root   = "./val_dataset"
    test_root  = "./test_dataset"
    sample_rate = 16000
    n_mels = 64
    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセット / ローダ
    train_ds = DirectoryAudioDataset(train_root, sample_rate=sample_rate, n_mels=n_mels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if os.path.isdir(val_root):
        val_ds = DirectoryAudioDataset(val_root, sample_rate=sample_rate, n_mels=n_mels)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_classes = len(train_ds.class_to_idx)
    model = ConvNet(n_mels=n_mels, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"epoch {epoch:03d} loss={avg_loss:.4f} acc={avg_acc:.4f}")

        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    v_logits = model(vx)
                    v_preds = v_logits.argmax(dim=1)
                    v_correct += (v_preds == vy).sum().item()
                    v_total += vx.size(0)
            print(f"  val acc = {v_correct / v_total:.4f}")

if __name__ == "__main__":
    main()
