# scripts/preprocess_train.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

DATA_PATH = "data/ecg5000.csv"
OUTDIR = "data/processed"
os.makedirs(OUTDIR, exist_ok=True)

print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH, header=None)

# Simulate label column if not present
if df.shape[1] == 141:
    df["label"] = np.random.randint(0, 2, size=len(df))

X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values.astype(int)

classes = np.unique(y)
label_map = {v: i for i, v in enumerate(classes)}
y = np.array([label_map[v] for v in y])

print(f"🔹 Found {len(classes)} classes: {label_map}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

np.save(os.path.join(OUTDIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTDIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTDIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTDIR, "y_val.npy"), y_val)
joblib.dump(scaler, os.path.join(OUTDIR, "scaler.pkl"))
joblib.dump(label_map, os.path.join(OUTDIR, "label_map.pkl"))

print("✅ Preprocessing complete! Saved data in:", OUTDIR)
