
# scripts/train_ecg.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, joblib, os

# Load processed data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")
label_map = joblib.load("data/processed/label_map.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("int64")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx][None, :], self.y[idx]

train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=64)

# Define CNN Model
class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

model = TinyCNN(n_classes=len(label_map)).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader.dataset):.4f} | Val Acc: {acc:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/ecg_cnn.pth")
joblib.dump(label_map, "models/label_map.pkl")
print("✅ Model training complete! Model saved to models/ecg_cnn.pth")
