# FrostGuardian
AI Edge Disease Detection for Cold Environments — prototype


This repo contains a minimal, end-to-end prototype: training scripts, TFLite conversion, a device-side inference example (Raspberry Pi / Linux), a tiny backend, and a simple frontend stub.


## Quick start 
1. Install device dependencies: `pip install -r device/requirements.txt`
2. Run: `python device/demo_record_and_infer.py`


## Contents
- model/: training, preprocessing, conversion to TensorFlow Lite
- device/: on-device inference and demo recorder
- backend/: simple Node.js server to accept results (optional)
- frontend/: small React app showing results (stub)


## License
MIT


### Devpost.md 




**Title:** FrostGuardian — AI Edge Disease Detection for Cold Environments


**Short description:** Portable edge device that analyzes cough/breath audio and skin temperature to provide early warning of respiratory illness. Works fully offline using TinyML and syncs results when connectivity becomes available.




## model/train.py


```python
"""
Minimal training script: loads precomputed mel-spectrogram features and trains a tiny CNN.
This uses PyTorch for convenience; later we export via ONNX -> TensorFlow -> TFLite or reimplement training in TF.


Assumes you have `X.npy` and `y.npy` prepared (features and labels).
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# hyperparams
BATCH = 32
EPOCHS = 10
LR = 1e-3


# toy dataset loader (replace with real data path)
X = np.load('X.npy') # shape (N, 1, H, W)
y = np.load('y.npy') # shape (N,)


X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)


loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, shuffle=True)


class TinyCNN(nn.Module):
def __init__(self, n_classes=3):
super().__init__()
self.net = nn.Sequential(
nn.Conv2d(1, 8, 3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(8, 16, 3, padding=1),
nn.ReLU(),
nn.AdaptiveAvgPool2d(1),
nn.Flatten(),
nn.Linear(16, n_classes)
)
def forward(self, x):
return self.net(x)


model = TinyCNN(n_classes=len(np.unique(y)))
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()


for e in range(EPOCHS):
total_loss = 0
for xb, yb in loader:
out = model(xb)
loss = crit(out, yb)
opt.zero_grad()
loss.backward()
opt.step()
total_loss += loss.item()
print(f"Epoch {e+1}/{EPOCHS} loss={total_loss/len(loader):.4f}")


# Save PyTorch model 
torch.save(model.state_dict(), 'model.pt')
print('Saved model.pt')
