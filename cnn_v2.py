#!/usr/bin/env python3
import torch
from torch import nn

ANGLE_CLASSES = list(range(-90, 91, 15))
NUM_ANGLE_CLASSES = len(ANGLE_CLASSES)

def angle_to_class(angle_deg):
    angle_deg = float(angle_deg)
    closest_idx = min(range(len(ANGLE_CLASSES)), key=lambda i: abs(ANGLE_CLASSES[i] - angle_deg))
    return closest_idx

def class_to_angle(class_idx):
    return ANGLE_CLASSES[int(class_idx)]

class SpeakerDirectionCNN(nn.Module):
    def __init__(self, 
                 n_mels=64,
                 n_angle_classes=NUM_ANGLE_CLASSES,
                 size_ch1=16,
                 size_ch2=32,
                 size_ch3=64,
                 size_kernel=3):
        super().__init__()

        self.n_angle_classes = n_angle_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, size_ch1, size_kernel, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(size_ch1, size_ch2, size_kernel, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(size_ch2, size_ch3, size_kernel, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(size_ch3, n_angle_classes)

    def forward(self, x):
        x = self.conv_layers(x)  
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    print(f"Angle classes: {ANGLE_CLASSES}")
    print(f"Number of angle classes: {NUM_ANGLE_CLASSES}")

    model1 = SpeakerDirectionCNN(n_mels=64, n_angle_classes=NUM_ANGLE_CLASSES)
    x = torch.randn(20, 1, 64, 16000)
    
    y1 = model1(x)   # [20, 13]
    print(f"SpeakerDirectionCNN output shape: {y1.shape}")

    pred_class = y1[0].argmax().item()
    pred_angle = class_to_angle(pred_class)
    print(f"Predicted class: {pred_class}, Angle: {pred_angle}°")
    