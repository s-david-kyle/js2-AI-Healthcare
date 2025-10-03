# clinicalbert_training.py
import os, csv, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clinicalbert_dataset import (
    ClinicalBERTFastDatasetWithIDs, collate_fn,
    ClinicalBERTPrecomputedDataset, collate_precomputed
)
from transformers import AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast
from resource_logger import ResourceLogger
from tqdm import tqdm
import argparse

# ----------------------------- Args / Seeds -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
parser.add_argument("--low_mem_mode", action="store_true", help="Enable low memory mode")
parser.add_argument("--precomputed", action="store_true", help="Use precomputed embeddings if available")
parser.add_argument("--precomputed_path", type=str, default=None, help="Path to precomputed npz file")
args = parser.parse_args()

METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")
PRECOMP_DEFAULT = f"./precomputed_bert_cls_{METRIC_PREFIX}.npz"
PRECOMP_PATH = args.precomputed_path or PRECOMP_DEFAULT
USE_PRECOMPUTED = args.precomputed or os.path.exists(PRECOMP_PATH)
print(f"🧩 Precomputed mode: {USE_PRECOMPUTED} | path: {PRECOMP_PATH if USE_PRECOMPUTED else 'N/A'}")

BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED); torch.manual_seed(SEED)
MAX_VISITS = 10

# ----------------------------- Load Data --------------------------------
BASE = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Structured sequences
X_train = np.load(f"{BASE}/X_train_transformer.npy")
y_train = np.load(f"{BASE}/y_train_transformer.npy")
m_train = np.load(f"{BASE}/mask_train_transformer.npy")
sid_train = np.load(f"{BASE}/subject_ids_train_transformer.npy")
X_val   = np.load(f"{BASE}/X_val_transformer.npy")
y_val   = np.load(f"{BASE}/y_val_transformer.npy")
m_val   = np.load(f"{BASE}/mask_val_transformer.npy")
sid_val = np.load(f"{BASE}/subject_ids_val_transformer.npy")

# Notes or embeddings
if USE_PRECOMPUTED:
    pre = np.load(PRECOMP_PATH)
    BERT_EMB_ALL = pre["embeddings"]
    sid_notes = pre["subject_ids"]
    print(f"📥 Loaded precomputed embeddings: {BERT_EMB_ALL.shape}")
else:
    X_notes = np.load(f"{BASE}/tokenized_input_ids_{METRIC_PREFIX}.npy")
    M_notes = np.load(f"{BASE}/tokenized_attention_masks_{METRIC_PREFIX}.npy")
    sid_notes = np.load(f"{BASE}/tokenized_subject_ids_{METRIC_PREFIX}.npy")
    TRUNC_TOKENS = int(os.getenv("TRUNC_TOKENS", 128))
    if X_notes.shape[-1] > TRUNC_TOKENS:
        X_notes = X_notes[:, :, :TRUNC_TOKENS]
        M_notes = M_notes[:, :, :TRUNC_TOKENS]
        print(f"✂️ Truncated note sequences to {TRUNC_TOKENS} tokens per visit.")

# Align
sid_to_idx = {int(s): i for i, s in enumerate(sid_notes)}
def align_split(X, y, m, sids):
    keep_mask = np.isin(sids, sid_notes)
    sids_keep = sids[keep_mask]
    X, y, m = X[keep_mask], y[keep_mask], m[keep_mask]
    note_idx = [sid_to_idx[int(s)] for s in sids_keep]
    if USE_PRECOMPUTED:
        emb = BERT_EMB_ALL[note_idx]
        return X, y, m, sids_keep, emb, None
    else:
        return X, y, m, sids_keep, X_notes[note_idx], M_notes[note_idx]

X_train, y_train, m_train, sid_train, A_train, B_train = align_split(X_train, y_train, m_train, sid_train)
X_val,   y_val,   m_val,   sid_val,   A_val,   B_val   = align_split(X_val,   y_val,   m_val,   sid_val)

if USE_PRECOMPUTED:
    print(f"📊 Train embs={A_train.shape}, Val embs={A_val.shape}")
else:
    print(f"📊 Train notes={A_train.shape}, Val notes={A_val.shape}")

# ----------------------------- Class Weights ----------------------------
class_counts = np.bincount(y_train.astype(int), minlength=3)
inv_freq = np.where(class_counts > 0, (len(y_train) / (3.0 * class_counts)), 0.0)
weight_tensor = torch.tensor(inv_freq, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# ----------------------------- DataLoaders ------------------------------
BATCH_SIZE  = 16 if (args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1") else 32
NUM_WORKERS = 2 if (args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1") else min(8, os.cpu_count() or 8)

if USE_PRECOMPUTED:
    train_ds = ClinicalBERTPrecomputedDataset(X_train, y_train, A_train, m_train, sid_train, max_visits=MAX_VISITS)
    val_ds   = ClinicalBERTPrecomputedDataset(X_val,   y_val,   A_val,   m_val,   sid_val,   max_visits=MAX_VISITS)
    _collate = collate_precomputed
else:
    train_ds = ClinicalBERTFastDatasetWithIDs(X_train, y_train, A_train, B_train, m_train, sid_train, max_visits=MAX_VISITS)
    val_ds   = ClinicalBERTFastDatasetWithIDs(X_val,   y_val,   A_val,   B_val,   m_val,   sid_val,   max_visits=MAX_VISITS)
    _collate = collate_fn

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))

# ----------------------------- Models ----------------------------------
from clinicalbert_model import ClinicalBERT_Transformer

class EmbeddingVisitTransformer(nn.Module):
    def __init__(self, structured_input_dim, hidden_dim=128, nhead=8, num_layers=2, dropout=0.4):
        super().__init__()
        self.bert_proj = nn.Linear(768, hidden_dim)
        self.struct_proj = nn.Linear(structured_input_dim, hidden_dim) if structured_input_dim is not None else None
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim) if self.struct_proj is not None else None
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
    def forward(self, bert_embs, structured_seq=None, visit_mask=None):
        x = self.bert_proj(bert_embs)
        if self.struct_proj is not None and structured_seq is not None:
            s = self.struct_proj(structured_seq)
            x = self.fusion_proj(torch.cat([x, s], dim=-1))
        x = self.encoder(x)
        if visit_mask is not None:
            m = visit_mask.unsqueeze(-1).float()
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else: pooled = x.mean(dim=1)
        return self.classifier(pooled)

if USE_PRECOMPUTED:
    model = EmbeddingVisitTransformer(structured_input_dim=X_train.shape[2]).to(device)
    print("🧠 Using EmbeddingVisitTransformer (no BERT in training).")
else:
    bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    for p in bert.parameters(): p.requires_grad = False
    for p in bert.encoder.layer[-2:].parameters(): p.requires_grad = True
    model = ClinicalBERT_Transformer(bert_model=bert, structured_input_dim=X_train.shape[2],
                                     hidden_dim=128, nhead=8, num_layers=2, dropout=0.4).to(device)
    print("🧊 Using ClinicalBERT_Transformer (fine-tuning last 2 layers).")

# ----------------------------- Optimizer/Training -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
num_epochs = 20
num_training_steps = max(1, len(train_loader)) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
scaler = GradScaler(enabled=torch.cuda.is_available())

BEST_PATH = f"{BASE}/clinicalbert_transformer_model_{METRIC_PREFIX}.pt"
METRIC_CSV = f"{BASE}/clinicalbert_transformer_metrics_{METRIC_PREFIX}.csv"
best_val_loss, patience, counter = float('inf'), 5, 0

# ----------------------------- Training Loop ----------------------------
try:
    with ResourceLogger(tag="clinicalbert_transformer"):
        for epoch in range(num_epochs):
            model.train(); train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                optimizer.zero_grad(set_to_none=True)
                if USE_PRECOMPUTED:
                    bert_embs, struct_seq, visit_mask, labels, _ = batch
                    bert_embs, struct_seq, visit_mask, labels = (
                        bert_embs.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                    )
                    with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                        logits = model(bert_embs, struct_seq, visit_mask)
                        loss = criterion(logits, labels)
                else:
                    ids, amask, struct_seq, visit_mask, labels, _ = batch
                    ids, amask, struct_seq, visit_mask, labels = (
                        ids.to(device), amask.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                    )
                    with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                        logits = model(ids, amask, struct_seq, visit_mask)
                        loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update(); scheduler.step()
                train_loss += loss.item()
            train_loss /= max(1, len(train_loader))
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

            # Validation
            model.eval(); val_loss, preds, trues, subj_out = 0.0, [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    if USE_PRECOMPUTED:
                        bert_embs, struct_seq, visit_mask, labels, subj_ids = batch
                        bert_embs, struct_seq, visit_mask, labels = (
                            bert_embs.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                        )
                        logits = model(bert_embs, struct_seq, visit_mask)
                    else:
                        ids, amask, struct_seq, visit_mask, labels, subj_ids = batch
                        ids, amask, struct_seq, visit_mask, labels = (
                            ids.to(device), amask.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                        )
                        logits = model(ids, amask, struct_seq, visit_mask)
                    loss = criterion(logits, labels); val_loss += loss.item()
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds.extend(np.argmax(probs, axis=1))
                    trues.extend(labels.cpu().numpy())
                    subj_out.extend(subj_ids.cpu().numpy())
            val_loss /= max(1, len(val_loader))
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, counter = val_loss, 0
                torch.save(model.state_dict(), BEST_PATH); print(f"  ✅ Saved best model at epoch {epoch+1}")
            else:
                counter += 1
                if counter >= patience: print("⏹️ Early stopping."); break
except Exception as e:
    print(f"Training crashed: {e}")

# ----------------------------- Save Metrics -----------------------------
print(f"\n💾 Saving predicted probabilities for {METRIC_PREFIX}...")
if os.path.exists(BEST_PATH):
    model.load_state_dict(torch.load(BEST_PATH, map_location=device)); model.eval()
    probs, preds, trues, subj_out = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving Probs"):
            if USE_PRECOMPUTED:
                bert_embs, struct_seq, visit_mask, labels, subj_ids = batch
                bert_embs, struct_seq, visit_mask, labels = (
                    bert_embs.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                )
                logits = model(bert_embs, struct_seq, visit_mask)
            else:
                ids, amask, struct_seq, visit_mask, labels, subj_ids = batch
                ids, amask, struct_seq, visit_mask, labels = (
                    ids.to(device), amask.to(device), struct_seq.to(device), visit_mask.float().to(device), labels.to(device)
                )
                logits = model(ids, amask, struct_seq, visit_mask)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.extend(p); preds.extend(np.argmax(p, axis=1))
            trues.extend(labels.cpu().numpy()); subj_out.extend(subj_ids.cpu().numpy())
    probs, preds, trues, subj_out = np.array(probs), np.array(preds), np.array(trues), np.array(subj_out)
    all_classes = [0,1,2]
    y_bin = label_binarize(trues, classes=all_classes)
    macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    micro_auc = roc_auc_score(y_bin, probs, average="micro", multi_class="ovr")
    print(f"ROC-AUC (macro): {macro_auc:.4f} | (micro): {micro_auc:.4f}")
    report = classification_report(trues, preds, labels=all_classes, zero_division=0, output_dict=True)
    with open(METRIC_CSV,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Class","Precision","Recall","F1-score"])
        for cls in all_classes:
            row=report.get(str(cls),{"precision":0,"recall":0,"f1-score":0})
            w.writerow([cls,f"{row['precision']:.4f}",f"{row['recall']:.4f}",f"{row['f1-score']:.4f}"])
        w.writerow(["Accuracy",f"{report.get('accuracy',0):.4f}","",""])
        w.writerow(["MacroAUC",f"{macro_auc:.4f}","",""]); w.writerow(["MicroAUC",f"{micro_auc:.4f}","",""])
    cm = confusion_matrix(trues, preds, labels=all_classes)
    plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
        xticklabels=all_classes,yticklabels=all_classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(f"{BASE}/clinicalbert_confusion_{METRIC_PREFIX}.png"); plt.close()
    np.savez_compressed(f"{BASE}/clinicalbert_transformer_probs_{METRIC_PREFIX}.npz",
        probs=probs, y_true=trues.astype(np.int64), subject_ids=subj_out.astype(np.int64))
    print(f"📦 Saved outputs → clinicalbert_transformer_probs_{METRIC_PREFIX}.npz")
else:
    print(f"❌ No saved model found at {BEST_PATH}")

del model; torch.cuda.empty_cache(); gc.collect()
