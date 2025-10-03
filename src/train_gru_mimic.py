"""train_gru_mimic.py
GRU baseline with shared validation IDs.
Produces metrics, plots, and stacking-ready outputs.
"""
import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from resource_logger import ResourceLogger
from tqdm import tqdm
import argparse

# ---------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# 1. Dataset & Model
# ---------------------------------------------------------------------
class SeqDS(Dataset):
    def __init__(self, X, y, sids):
        self.X, self.y, self.sids = X, y, sids
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(int(self.y[i]), dtype=torch.long),
            torch.tensor(int(self.sids[i]), dtype=torch.long),
        )

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, dropout=0.3, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ---------------------------------------------------------------------
# 2. CLI args & paths
# ---------------------------------------------------------------------
BASE = "./"
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix

# ---------------------------------------------------------------------
# 3. Load sequences & splits
# ---------------------------------------------------------------------
X_train = np.load(f"{BASE}/X_train_transformer.npy")
y_train = np.load(f"{BASE}/y_train_transformer.npy")
sid_train = np.load(f"{BASE}/subject_ids_train_transformer.npy")

X_val = np.load(f"{BASE}/X_val_transformer.npy")
y_val = np.load(f"{BASE}/y_val_transformer.npy")
sid_val = np.load(f"{BASE}/subject_ids_val_transformer.npy")

print(f"Train size: {len(y_train)} | Val size: {len(y_val)}")

# ---------------------------------------------------------------------
# 4. DataLoaders
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(SeqDS(X_train, y_train, sid_train), batch_size=32,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(SeqDS(X_val, y_val, sid_val), batch_size=32,
                        shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------
# 5. Model, optimizer, loss
# ---------------------------------------------------------------------
model = GRUClassifier(input_dim=X_train.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Class weights (balanced)
class_counts = np.bincount(y_train)
class_weights = 1.0 / np.maximum(class_counts, 1)
norm_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(norm_weights, dtype=torch.float32).to(device)
crit = nn.CrossEntropyLoss(weight=weight_tensor)

BEST = f"{BASE}/gru_model_{METRIC_PREFIX}.pt"
METRIC = f"{BASE}/gru_metrics_{METRIC_PREFIX}.csv"

# ---------------------------------------------------------------------
# 6. Training loop with early stopping
# ---------------------------------------------------------------------
with ResourceLogger(tag=f"gru_multiclass_{METRIC_PREFIX}"):
    best, pat, cnt = float("inf"), 3, 0
    for ep in range(20):
        # --- Train ---
        model.train(); tl = 0
        for xb, yb, _ in tqdm(train_loader, desc=f"Epoch {ep+1:02d} Training"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            tl += loss.item()
        print(f"Ep{ep+1:02d} Train {tl/len(train_loader):.4f}")

        # --- Validate ---
        model.eval(); vl=0; preds=[]; trues=[]; subj_out=[]; all_probs=[]
        with torch.no_grad():
            for xb, yb, sids in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                vl += crit(logit, yb).item()
                prob_batch = torch.softmax(logit, dim=1).cpu().numpy()
                preds.extend(np.argmax(prob_batch, axis=1))
                trues.extend(yb.cpu().numpy())
                subj_out.extend(sids.cpu().numpy())
                all_probs.extend(prob_batch)
        vl /= len(val_loader)
        print(f"Ep{ep+1:02d} Val {vl:.4f}")

        if vl < best:
            best = vl; cnt = 0
            torch.save(model.state_dict(), BEST)
            print("  ✓ checkpoint saved")
        else:
            cnt += 1
            if cnt >= pat:
                print("Early stopping")
                break

# ---------------------------------------------------------------------
# 7. Final Evaluation
# ---------------------------------------------------------------------
model.load_state_dict(torch.load(BEST))
model.eval()
preds, trues, all_probs, subj_out = [], [], [], []
with torch.no_grad():
    for xb, yb, sids in val_loader:
        xb = xb.to(device)
        logit = model(xb)
        prob_batch = torch.softmax(logit, dim=1).cpu().numpy()
        preds.extend(np.argmax(prob_batch, axis=1))
        trues.extend(yb.numpy())
        subj_out.extend(sids.numpy())
        all_probs.extend(prob_batch)

preds, trues = np.array(preds), np.array(trues)
probs = np.array(all_probs)
subj_out = np.array(subj_out)

# Metrics
acc = accuracy_score(trues, preds)
report = classification_report(trues, preds, labels=[0,1,2], zero_division=0, output_dict=True)

# ROC–AUC
y_bin = label_binarize(trues, classes=[0,1,2])
macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
micro_auc = roc_auc_score(y_bin, probs, average="micro", multi_class="ovr")

# Save metrics CSV
with open(METRIC, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class","Precision","Recall","F1-score"])
    for cls in ["0","1","2"]:
        row = report.get(cls, {"precision":0,"recall":0,"f1-score":0})
        writer.writerow([cls,row["precision"],row["recall"],row["f1-score"]])
    writer.writerow(["Accuracy", acc,"",""])
    writer.writerow(["MacroAUC", macro_auc,"",""])
    writer.writerow(["MicroAUC", micro_auc,"",""])
print(f"Metrics → {METRIC}")

# Confusion Matrix
cm = confusion_matrix(trues, preds, labels=[0,1,2])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix - GRU")
plt.tight_layout()
plt.savefig(f"{BASE}/gru_confusion_{METRIC_PREFIX}.png")
plt.close()

# ROC Curves
plt.figure()
for i, cls in enumerate([0,1,2]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    auc_i = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_i:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("GRU Multiclass ROC")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{BASE}/gru_roc_{METRIC_PREFIX}.png")
plt.close()

# ---------------------------------------------------------------------
# 8. Save probabilities for stacking
# ---------------------------------------------------------------------
np.savez_compressed(
    f"{BASE}/gru_probs_{METRIC_PREFIX}.npz",
    probs=probs,
    y_true=trues,
    subject_ids=subj_out
)
print(f"Saved GRU probs → gru_probs_{METRIC_PREFIX}.npz")
print(f"✅ Finished GRU training with shared val IDs [{METRIC_PREFIX}]")
