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
from resource_logger import ResourceLogger
from clinicalbert_dataset import ClinicalBERTDataset
from clinicalbert_model import ClinicalBERT_Transformer  # Updated model filename
from tqdm import tqdm
import argparse
import gc
import pynvml

parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
parser.add_argument("--low_mem_mode", action="store_true", help="Enable low memory mode")
args = parser.parse_args()

METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")

BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)
torch.manual_seed(SEED)

def collate_fn(batch):
    """
    Pads tokenized note sequences and structured features for batching.

    Each batch item is:
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
    labels = [item[3] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True)  # pads T dimension
    padded_attention = pad_sequence(attention_masks, batch_first=True)
    padded_structured = pad_sequence(structured, batch_first=True)

    # Ensure labels are int tensors
    labels = torch.tensor([int(l) if not torch.is_tensor(l) else l.item() for l in labels], dtype=torch.long)

    return padded_input_ids, padded_attention, padded_structured, labels

# Loading your data (replace with actual loading code)
BASE = "./"
shared_val_ids = np.load(f"{BASE}/shared_val_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
X_all = np.load(f"{BASE}/X_seq.npy")
y_all = np.load(f"{BASE}/y_seq.npy")
subject_ids = np.load(f"{BASE}/subject_ids_seq.npy")
val_subject_ids = subject_ids[np.isin(subject_ids, shared_val_ids)]

print("Number of shared_val_ids:", len(shared_val_ids))

mask = np.isin(subject_ids, shared_val_ids)
X_train, X_val = X_all[~mask], X_all[mask]
y_train, y_val = y_all[~mask], y_all[mask]
notes = torch.load(f"{BASE}/tokenized_clinicalbert_notes.pt", weights_only=False)

print(f"  Validation samples: {len(y_val)}")

val_subject_ids = subject_ids[np.isin(subject_ids, shared_val_ids)]
notes_dict = {int(k): v for k, v in notes.items()}
valid_ids = [sid for sid in val_subject_ids if sid in notes_dict]

mask = np.isin(val_subject_ids, valid_ids)
X_val = X_val[mask]
y_val = y_val[mask]
val_subject_ids = val_subject_ids[mask]
notes = [notes_dict[sid] for sid in val_subject_ids]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n?? Dataset Split Summary:")
print(f"  Training samples:   {len(y_train)}")
print(f"  Validation samples: {len(y_val)}")

# Class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
norm_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(norm_weights, dtype=torch.float32).to(device)

train_ds = ClinicalBERTDataset(X_train, y_train, notes)
val_ds = ClinicalBERTDataset(X_val, y_val, notes)

BATCH_SIZE = 16 if (args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1") else 32
NUM_WORKERS = 2 if (args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1") else 4
PIN_MEMORY = not (args.low_mem_mode or os.getenv("LOW_MEM_MODE") == "1")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = ClinicalBERT_Transformer(bert_model=bert,
                                 structured_input_dim=X_train.shape[2],
                                 hidden_dim=128,
                                 nhead=8,
                                 num_layers=2,
                                 dropout=0.4)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

num_epochs = 20
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

#scaler = torch.cuda.amp.GradScaler(device='cuda')
scaler = GradScaler()

#criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1])  # binary classification
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

BEST_PATH = f"{BASE}/clinicalbert_transformer_model_{METRIC_PREFIX}.pt"
METRIC_CSV = f"{BASE}/clinicalbert_transformer_metrics_{METRIC_PREFIX}.csv"

best_val_loss = float('inf')
patience = 5
counter = 0

try:
    with ResourceLogger(tag="clinicalbert_transformer"):
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for input_ids, attention_mask, struct_seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                struct_seq = struct_seq.to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    logits = model(input_ids, attention_mask, struct_seq)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0
            preds = []
            trues = []

            with torch.no_grad():
                for input_ids, attention_mask, struct_seq, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    struct_seq = struct_seq.to(device)
                    labels = labels.long().to(device)

                    logits = model(input_ids, attention_mask, struct_seq)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 3)
                    batch_preds = np.argmax(probs, axis=1)
                    #probs = torch.sigmoid(logits).cpu().numpy()
                    #batch_preds = (probs >= 0.5).astype(int)
                    preds.extend(batch_preds)
                    trues.extend(labels.cpu().numpy().astype(int))

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), BEST_PATH)
                print(f"  Saved best model at epoch {epoch+1}")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

except Exception as e:
    print("bert_emb:", bert_emb.shape)
    print("structured_seq:", structured_seq.shape)
    print(f"Training crashed: {e}")

# ---------------------------------------------------------------------
# Save prediction probabilities for stacking
# ---------------------------------------------------------------------
finally:
    print(f"\n?? Saving predicted probabilities for {METRIC_PREFIX}...")

    model = ClinicalBERT_Transformer(bert_model=bert,
                                     structured_input_dim=X_train.shape[2],
                                     hidden_dim=128,
                                     nhead=8,
                                     num_layers=2,
                                     dropout=0.4).to(device)

    model.load_state_dict(torch.load(BEST_PATH))
    model.eval()

    probs, preds, trues = [], [], []

    with torch.no_grad():
        for ids, mask, struct, lbl in tqdm(val_loader, desc="Saving Probs"):
            ids = ids.to(device)
            mask = mask.to(device)
            struct = struct.to(device)
            lbl = lbl.to(device).float()

            logits = model(ids, mask, struct)  # logits: [B, C]
            prob = torch.softmax(logits, dim=1).cpu().numpy()  # [B, C]
            pred = np.argmax(prob, axis=1)  # [B]
            true = lbl.cpu().numpy()        # [B]
            
            probs.extend(prob)
            preds.extend(pred)
            trues.extend(true)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("?? Cleaned up model and freed GPU memory.")

# ?? Reload model from checkpoint (optional redundancy)
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = ClinicalBERT_Transformer(bert_model=bert,
                                 structured_input_dim=X_train.shape[2],
                                 hidden_dim=128,
                                 nhead=8,
                                 num_layers=2,
                                 dropout=0.4).to(device)
model.load_state_dict(torch.load(BEST_PATH))
model.eval()

# Convert to NumPy arrays
probs_np = np.array(probs)
y_val_np = np.array(y_val)
subj_out = val_subject_ids

# ?? Sanity checks
print(f"? probs shape:      {probs_np.shape}")
print(f"? y_val shape:      {y_val_np.shape}")
print(f"? subject_ids shape:{subj_out.shape}")
assert probs_np.shape[0] == y_val_np.shape[0] == subj_out.shape[0], (
    f"? Mismatch in number of samples: probs {probs_np.shape[0]}, "
    f"labels {y_val_np.shape[0]}, subject_ids {subj_out.shape[0]}"
)

# ---------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------
from sklearn.utils.multiclass import unique_labels

# Explicitly define your full set of labels
all_classes = [0, 1, 2]

# Generate classification report
report = classification_report(
    trues, preds,
    labels=all_classes,
    zero_division=0,
    output_dict=True
)

# Save to CSV
with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score"])
    
    for cls in all_classes:
        cls_str = str(cls)
        if cls_str in report:
            row = report[cls_str]
            writer.writerow([
                cls_str,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}"
            ])
        else:
            # Class was missing in predictions or ground truth
            writer.writerow([cls_str, "0.0000", "0.0000", "0.0000"])

# Print full report to console
print(classification_report(trues, preds, labels=all_classes, zero_division=0))

'''
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

print(f"?? Metrics ? {METRIC_CSV}")
print(classification_report(trues, preds, zero_division=0))
'''

# ---------------------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - ClinicalBERT+Transformer")
plt.tight_layout()
conf_matrix_path = f"{BASE}/clinicalbert_confusion_{METRIC_PREFIX}.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"??? Confusion matrix saved ? {conf_matrix_path}")

# ---------------------------------------------------------------------
# Save probabilities for stacking
# ---------------------------------------------------------------------
npz_path = f"{BASE}/clinicalbert_transformer_probs_{METRIC_PREFIX}.npz"
np.savez_compressed(
    npz_path,
    probs=probs_np,
    y_true=y_val_np,
    subject_ids=subj_out
)
print(f"?? Saved ClinicalBERT+Transformer probs ? {npz_path}")

if probs_np.shape[0] == 0:
    print("?? No validation samples were processed ? empty .npz file!")
   