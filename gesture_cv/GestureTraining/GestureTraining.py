import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#get into /GestureTraining and run python3 GestureTraining.py

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../GestureData/gesture_data_full.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../GestureData/gesture_classifier.pt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "../GestureData/label_map.json")

# -------------------------
# Load CSV
# -------------------------
Xs = []
ys = []

with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        *features, label = row
        Xs.append([float(v) for v in features])
        ys.append(label)

X = np.asarray(Xs, dtype=np.float32)  # (N, 63)
y_text = np.asarray(ys)               # (N,)

# -------------------------
# Build label index mapping
# -------------------------
labels = sorted(set(y_text.tolist()))
label_to_index = {lab: i for i, lab in enumerate(labels)}
num_classes = len(labels)

y = np.array([label_to_index[lab] for lab in y_text], dtype=np.int64)

print(f"Loaded {len(X)} samples, {num_classes} gesture classes.")
print("Classes:", labels)

# Save label list for inference
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(labels, f)
print(f"Saved label map to {LABEL_MAP_PATH}")

# -------------------------
# Train/val split (manual, no sklearn)
# -------------------------
rng = np.random.default_rng(seed=42)
indices = np.arange(len(X))
rng.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

X_train = torch.tensor(X[train_idx], dtype=torch.float32)
y_train = torch.tensor(y[train_idx], dtype=torch.long)
X_val = torch.tensor(X[val_idx], dtype=torch.float32)
y_val = torch.tensor(y[val_idx], dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# -------------------------
# Model definition
# -------------------------
class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = GestureNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Training loop
# -------------------------
EPOCHS = 40

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    model.train()
    return correct / max(total, 1)

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_ds)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.3f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved PyTorch model to {MODEL_PATH}")
