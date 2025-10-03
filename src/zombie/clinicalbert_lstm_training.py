"""clinicalbert_lstm_training.py
Clean end‑to‑end script that
1. Loads tokenized note sequences + structured features
2. Trains ClinicalBERT + LSTM with early stopping
3. Logs runtime / GPU hours / disk via ResourceLogger
4. Writes metrics (AUC, accuracy) and ROC plot
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from resource_logger import ResourceLogger
from clinicalbert_lstm_dataset import ClinicalBERTLSTMDataset
from clinicalbert_lstm_model import ClinicalBERT_LSTM
from tqdm import tqdm
import argparse
import gc
import pynvml

# ---------------------------------------------------------------------
# Optional: Use command-line or environment argument to control prefix
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
parser.add_argument("--low_mem_mode", action="store_true", help="Enable low memory mode")
args = parser.parse_args()

# Allow fallback to environment variable
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")

# Base seed
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET

np.random.seed(SEED)
torch.manual_seed(SEED)

def collate_fn(batch):
    """
    Pads tokenized note sequences and structured features for batching.

    Each batch entry is:
      - input_ids:      [T, L]
      - attention_mask: [T, L]
      - structured:     [T, F]
      - label:          scalar

    Returns:
      - input_ids:      [B, T, L]
      - attention_mask: [B, T, L]
      - structured:     [B, T, F]
      - labels:         [B]
    """
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    structured = [item[2] for item in batch]
    visit_masks = [item[3] for item in batch]
    labels = [item[4] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True)
    padded_attention = pad_sequence(attention_masks, batch_first=True)
    padded_structured = pad_sequence(structured, batch_first=True)
    padded_visit_masks = pad_sequence(visit_masks, batch_first=True)  # [B, T]

    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)

    return padded_input_ids, padded_attention, padded_structured, padded_visit_masks, labels


BASE = "./"
shared_val_ids = np.load(f"{BASE}/shared_val_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
X_all = np.load(f"{BASE}/X_seq.npy")
y_all = np.load(f"{BASE}/y_seq.npy")
subject_ids = np.load(f"{BASE}/subject_ids_seq.npy")
val_subject_ids = subject_ids[np.isin(subject_ids, shared_val_ids)]

mask = np.isin(subject_ids, shared_val_ids)
X_train, X_val = X_all[~mask], X_all[mask]
y_train, y_val = y_all[~mask], y_all[mask]
notes = torch.load(f"{BASE}/tokenized_clinicalbert_notes.pt", weights_only=False)

# Only keep tokenized notes for validation subjects
# Ensure alignment between structured validation data and tokenized notes
val_subject_ids = subject_ids[np.isin(subject_ids, shared_val_ids)]
notes_dict = {int(k): v for k, v in notes.items()}
valid_ids = [sid for sid in val_subject_ids if sid in notes_dict]

# Filter all validation data based on valid note availability
mask = np.isin(val_subject_ids, valid_ids)
X_val = X_val[mask]
y_val = y_val[mask]
val_subject_ids = val_subject_ids[mask]
notes = [notes_dict[sid] for sid in val_subject_ids]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
norm_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(norm_weights, dtype=torch.float32).to(device)

train_ds = ClinicalBERTLSTMDataset(X_train, y_train, notes)
val_ds = ClinicalBERTLSTMDataset(X_val, y_val, notes)
assert len(val_ds) == len(val_subject_ids), "❌ Dataset and subject ID mismatch!"

if args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1":
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    PIN_MEMORY = False
    print(f"🧠 Low memory mode enabled: BATCH_SIZE={BATCH_SIZE}, workers={NUM_WORKERS}")
else:
    # Try adaptive batch sizing based on available GPU memory
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        raw_free_gb = info.free / 1024**3
        free_gb = max(0, raw_free_gb - 6)  # Subtract safety buffer
        pynvml.nvmlShutdown()

        if free_gb < 4:
            BATCH_SIZE = 8
            NUM_WORKERS = 2
        elif free_gb < 10:
            BATCH_SIZE = 16
            NUM_WORKERS = 2
        else:
            BATCH_SIZE = 32
            NUM_WORKERS = 4


        PIN_MEMORY = True
        print(f"⚙️ Adaptive mode: {free_gb:.2f} GB free → BATCH_SIZE={BATCH_SIZE}, workers={NUM_WORKERS}")

    except Exception as e:
        BATCH_SIZE = 64
        NUM_WORKERS = 8
        PIN_MEMORY = True
        print(f"⚠️ pynvml unavailable. Defaulting to BATCH_SIZE={BATCH_SIZE}. Error: {e}")


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=collate_fn, num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)


print(f"🧮 Using batch_size = {BATCH_SIZE}")
print(f"# validation samples: {len(val_ds)}")

bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = ClinicalBERT_LSTM(bert, structured_input_dim=X_train.shape[2], hidden_dim=128,
                          lstm_layers=2, dropout=0.4, finetune_bert=False)
torch.cuda.empty_cache()
gc.collect()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

num_epochs = 20
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

scaler    = GradScaler(device='cuda')

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
BEST_PATH = f"{BASE}/clinicalbert_lstm_model_{METRIC_PREFIX}.pt"
METRIC_CSV = f"{BASE}/clinicalbert_lstm_metrics_{METRIC_PREFIX}.csv"

best_val, patience, counter = float('inf'), 5, 0
bert_unfrozen = False
bert_unfrozen_epoch = None
prev_val_loss = float('inf')
VAL_PLATEAU_THRESHOLD = 0.001
MIN_UNFREEZE_EPOCH = 2
accum_steps = 2
if device.type == "cuda":
    try:
        torch.cuda.set_per_process_memory_fraction(0.6, device=device.index or 0)
        print(f"⚠️ Using only 60% of GPU memory to avoid OOM.")
    except Exception as e:
        print(f"⚠️ Failed to set per-process memory fraction: {e}")


try:
    with ResourceLogger(tag="clinicalbert_lstm"):
        for ep in range(20):
            torch.cuda.empty_cache()
            gc.collect()
            model.train(); train_loss = 0
            for ids, mask, struct, lbl in tqdm(train_loader, desc=f"Epoch {ep+1:02d} Training"):
                ids, mask, struct, lbl = ids.to(device), mask.to(device), struct.to(device), lbl.to(device)
                optimizer.zero_grad()
        for i, (ids, mask, struct, lbl) in enumerate(tqdm(train_loader, desc=f"Epoch {ep+1:02d} Training")):
            ids, mask, struct, lbl = ids.to(device), mask.to(device), struct.to(device), lbl.to(device)

            with autocast(device_type="cuda"):
                logits = model(ids, mask, struct)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, lbl)
                loss = loss / accum_steps  # normalize loss

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # <-- Move AFTER optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            scheduler.step()
            print(f"Ep{ep+1:02d} Train {train_loss/len(train_loader):.4f}")

            model.eval(); val_loss = 0; preds = []; trues = []
            with torch.no_grad():
                for ids, mask, struct, lbl in tqdm(val_loader, desc=f"Epoch {ep+1:02d} Validation"):
                    ids, mask, struct, lbl = ids.to(device), mask.to(device), struct.to(device), lbl.to(device)
                    logits = model(ids, mask, struct)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    val_loss += criterion(logits, lbl).item()
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    trues.extend(lbl.cpu().numpy())

            val_loss /= len(val_loader)
            print(f"Ep{ep+1:02d} Val {val_loss:.4f}")

            if not bert_unfrozen and ep >= MIN_UNFREEZE_EPOCH and abs(prev_val_loss - val_loss) < VAL_PLATEAU_THRESHOLD:
                print(f"🔓 Unfreezing ClinicalBERT at epoch {ep+1} due to val loss plateau")
                for param in model.bert.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
                scheduler = CosineAnnealingLR(optimizer, T_max=20)
                bert_unfrozen = True
                bert_unfrozen_epoch = ep + 1

            prev_val_loss = val_loss

            if val_loss < best_val:
                best_val = val_loss; counter = 0
                torch.save(model.state_dict(), BEST_PATH)
                print("  ✓ checkpoint saved")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping"); break
except Exception as e:
    print(f"🔥 Training crashed: {e}")

# ---------------------------------------------------------------------
# Save prediction probabilities for stacking
# ---------------------------------------------------------------------
finally:
    print(f"\n📦 Saving predicted probabilities for {METRIC_PREFIX}...")
    model = ClinicalBERT_LSTM(bert, structured_input_dim=X_train.shape[2], hidden_dim=128,
                              lstm_layers=2, dropout=0.4, finetune_bert=False).to(device)
    model.load_state_dict(torch.load(BEST_PATH))
    model.eval()
    probs, preds, trues = [], [], []

    with torch.no_grad():
        for ids, mask, struct, lbl in tqdm(val_loader, desc="Saving Probs"):
            ...
            # prediction loop
            ...

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleaned up model and freed GPU memory.")

# 🧠 Reload model from checkpoint
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = ClinicalBERT_LSTM(bert, structured_input_dim=X_train.shape[2], hidden_dim=128,
                          lstm_layers=2, dropout=0.4, finetune_bert=False)
model.load_state_dict(torch.load(BEST_PATH))
model.to(device)
model.eval()

# Convert to NumPy arrays
probs_np = np.array(probs)
y_val_np = np.array(y_val)
subj_out = val_subject_ids

# 🔍 Sanity checks
print(f"✅ probs shape:      {probs_np.shape}")
print(f"✅ y_val shape:      {y_val_np.shape}")
print(f"✅ subject_ids shape:{subj_out.shape}")
assert probs_np.shape[0] == y_val_np.shape[0] == subj_out.shape[0], (
    f"❌ Mismatch in number of samples: probs {probs_np.shape[0]}, "
    f"labels {y_val_np.shape[0]}, subject_ids {subj_out.shape[0]}"
)

# ---------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------
acc = accuracy_score(trues, preds)
report = classification_report(trues, preds, zero_division=0, output_dict=True)

with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score"])
    for cls in sorted(report.keys()):
        if cls in ["accuracy"]:
            continue
        row = report[cls]
        writer.writerow([cls, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}"])

print(f"📊 Metrics → {METRIC_CSV}")
print(classification_report(trues, preds, zero_division=0))

# ---------------------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - ClinicalBERT+LSTM")
plt.tight_layout()
conf_matrix_path = f"{BASE}/clinicalbert_confusion_{METRIC_PREFIX}.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"🖼️ Confusion matrix saved → {conf_matrix_path}")

# ---------------------------------------------------------------------
# Save probabilities for stacking
# ---------------------------------------------------------------------
npz_path = f"{BASE}/clinicalbert_lstm_probs_{METRIC_PREFIX}.npz"
np.savez_compressed(
    npz_path,
    probs=probs_np,
    y_true=y_val_np,
    subject_ids=subj_out
)
print(f"📦 Saved ClinicalBERT+LSTM probs → {npz_path}")

if probs_np.shape[0] == 0:
    print("⚠️ No validation samples were processed — empty .npz file!")