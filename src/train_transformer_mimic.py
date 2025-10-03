"""train_transformer_mimic.py
Transformer baseline on structured sequences with shared validation IDs.
Now saves per-class ROC curves and stacking probabilities for meta-learning.
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
import argparse

# ------------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------
# 1. Dataset & Model
# ------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, seq, labels, masks, subj_ids):
        self.X, self.y, self.masks, self.sids = seq, labels, masks, subj_ids
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.sids[idx], dtype=torch.long)
        )

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, model_dim=64, heads=4, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(model_dim, num_classes)
    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=(mask==0) if mask is not None else None)
        return self.fc(x[:, -1, :])

# ------------------------------------------------------------------
# 2. CLI & Paths
# ------------------------------------------------------------------
BASE = "./"
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix

# ------------------------------------------------------------------
# 3. Load split files from transformer_sequences.py
# ------------------------------------------------------------------
X_train = np.load(f"{BASE}/X_train_transformer.npy")
y_train = np.load(f"{BASE}/y_train_transformer.npy")
m_train = np.load(f"{BASE}/mask_train_transformer.npy")
sid_train = np.load(f"{BASE}/subject_ids_train_transformer.npy")

X_val = np.load(f"{BASE}/X_val_transformer.npy")
y_val = np.load(f"{BASE}/y_val_transformer.npy")
m_val = np.load(f"{BASE}/mask_val_transformer.npy")
sid_val = np.load(f"{BASE}/subject_ids_val_transformer.npy")

print(f"Train size: {len(y_train)} | Val size: {len(y_val)}")

train_loader = DataLoader(SequenceDataset(X_train, y_train, m_train, sid_train),
                          batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(SequenceDataset(X_val,   y_val,   m_val,   sid_val),
                          batch_size=32, shuffle=False, num_workers=2)

# ------------------------------------------------------------------
# 4. Model / loss / opt
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim=X_train.shape[2], num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Balanced class weights
class_counts = np.bincount(y_train)
weights = len(y_train) / (len(class_counts) * class_counts)
weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

BEST_PATH   = f"{BASE}/mimic_transformer_model_{METRIC_PREFIX}.pt"
METRIC_CSV  = f"{BASE}/transformer_metrics_{METRIC_PREFIX}.csv"

# ------------------------------------------------------------------
# 5. Train + validate with ResourceLogger
# ------------------------------------------------------------------
with ResourceLogger(tag=f"transformer_{METRIC_PREFIX}"):
    best_val, patience, counter = float('inf'), 3, 0
    for epoch in range(20):
        # --- train ---
        model.train(); tr_loss=0
        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            logits = model(xb, mask=mb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | TrainLoss {tr_loss/len(train_loader):.4f}")

        # --- val ---
        model.eval(); val_loss=0; preds=[]; trues=[]; all_probs=[]; subj_out=[]
        with torch.no_grad():
            for xb, yb, mb, sids in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = model(xb, mask=mb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
                preds.extend(np.argmax(probs, axis=1))
                trues.extend(yb.cpu().numpy())
                subj_out.extend(sids.numpy())
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d} | ValLoss  {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss; counter=0
            torch.save(model.state_dict(), BEST_PATH)
            print("  ✓ checkpoint saved")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

# ------------------------------------------------------------------
# 6. Final Evaluation
# ------------------------------------------------------------------
model.load_state_dict(torch.load(BEST_PATH)); model.eval()
preds, trues, all_probs, subj_out = [], [], [], []
with torch.no_grad():
    for xb, yb, mb, sids in val_loader:
        xb, mb = xb.to(device), mb.to(device)
        logits = model(xb, mask=mb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        preds.extend(np.argmax(probs, axis=1))
        trues.extend(yb.numpy())
        subj_out.extend(sids.numpy())

preds, trues = np.array(preds), np.array(trues)
probs = np.array(all_probs)
subj_out = np.array(subj_out)
acc = accuracy_score(trues, preds)

# Classification report
report = classification_report(trues, preds, labels=[0,1,2], zero_division=0, output_dict=True)

# ROC–AUC
y_bin = label_binarize(trues, classes=[0,1,2])
macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
micro_auc = roc_auc_score(y_bin, probs, average="micro", multi_class="ovr")

# ------------------------------------------------------------------
# 7. Save metrics
# ------------------------------------------------------------------
with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class","Precision","Recall","F1-score"])
    for cls in ["0","1","2"]:
        row = report.get(cls, {"precision":0,"recall":0,"f1-score":0})
        writer.writerow([cls,row["precision"],row["recall"],row["f1-score"]])
    writer.writerow(["Accuracy", acc, "", ""])
    writer.writerow(["MacroAUC", macro_auc, "", ""])
    writer.writerow(["MicroAUC", micro_auc, "", ""])
print(f"Metrics → {METRIC_CSV}")

# ------------------------------------------------------------------
# 8. Confusion Matrix + ROC curves
# ------------------------------------------------------------------
cm = confusion_matrix(trues, preds, labels=[0,1,2])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Transformer Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{BASE}/transformer_confusion_{METRIC_PREFIX}.png")
plt.close()

plt.figure()
for i, cls in enumerate([0,1,2]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    auc_i = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_i:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Transformer Multiclass ROC")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{BASE}/transformer_roc_curve_{METRIC_PREFIX}.png")
plt.close()

# ------------------------------------------------------------------
# 9. Save stacking probabilities
# ------------------------------------------------------------------
np.savez_compressed(
    f"{BASE}/transformer_probs_{METRIC_PREFIX}.npz",
    probs=probs,
    y_true=trues,
    subject_ids=subj_out
)
print(f"Saved Transformer probs → transformer_probs_{METRIC_PREFIX}.npz")
print(f"✅ Finished Transformer training with split files [{METRIC_PREFIX}]")
