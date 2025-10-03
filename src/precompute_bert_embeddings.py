import os
import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm
import argparse

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=os.getenv("METRIC_PREFIX", "iter1"))
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
args = parser.parse_args()

BASE = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_PREFIX = args.metric_prefix

# ---------------- Load tokenized notes ----------------
X_notes = np.load(f"{BASE}/tokenized_input_ids_{METRIC_PREFIX}.npy")   # (N, T, L)
M_notes = np.load(f"{BASE}/tokenized_attention_masks_{METRIC_PREFIX}.npy")  # (N, T, L)
sid_notes = np.load(f"{BASE}/tokenized_subject_ids_{METRIC_PREFIX}.npy")

print("Notes:", X_notes.shape, "Masks:", M_notes.shape)

# Truncate for speed (optional)
TRUNC_TOKENS = int(os.getenv("TRUNC_TOKENS", 128))
if X_notes.shape[-1] > TRUNC_TOKENS:
    X_notes = X_notes[:, :, :TRUNC_TOKENS]
    M_notes = M_notes[:, :, :TRUNC_TOKENS]
    print(f"✂️ Truncated to {TRUNC_TOKENS} tokens per visit")

N, T, L = X_notes.shape
all_embeddings = np.zeros((N, T, 768), dtype=np.float32)

# ---------------- Paths ----------------
out_path = f"{BASE}/precomputed_bert_cls_{METRIC_PREFIX}.npz"
tmp_path = f"{BASE}/precomputed_bert_cls_{METRIC_PREFIX}_tmp.npz"

# ---------------- Resume logic ----------------
start_idx = 0
if args.resume and os.path.exists(tmp_path):
    tmp = np.load(tmp_path)
    all_embeddings = tmp["embeddings"]
    sid_tmp = tmp["subject_ids"]
    if not np.array_equal(sid_tmp, sid_notes):
        raise ValueError("❌ Subject IDs in checkpoint do not match current data.")
    nonzero_rows = np.where(all_embeddings.sum(axis=(1,2)) != 0)[0]
    if len(nonzero_rows) > 0:
        start_idx = nonzero_rows[-1] + 1
    print(f"🔄 Resuming from checkpoint at {tmp_path} (already processed {start_idx}/{N})")

# ---------------- Load ClinicalBERT ----------------
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
bert.eval()
for p in bert.parameters():
    p.requires_grad = False

# ---------------- Batched precompute ----------------
BATCH_SIZE = 16       # adjust to fit GPU memory
SAVE_EVERY = 1000     # checkpoint every 1000 patients

with torch.no_grad():
    for start in tqdm(range(start_idx, N, BATCH_SIZE), desc="Precomputing CLS embeddings"):
        end = min(start + BATCH_SIZE, N)
        batch_ids = torch.tensor(X_notes[start:end], dtype=torch.long, device=device)   # (B, T, L)
        batch_mask = torch.tensor(M_notes[start:end], dtype=torch.long, device=device)  # (B, T, L)

        B, T, L = batch_ids.shape
        flat_ids = batch_ids.view(B * T, L)       # (B*T, L)
        flat_mask = batch_mask.view(B * T, L)     # (B*T, L)

        out = bert(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        cls_emb = out[:, 0, :].view(B, T, 768).cpu().numpy()   # (B, T, 768)

        all_embeddings[start:end] = cls_emb

        # Periodic checkpoint
        if (start // BATCH_SIZE) % (SAVE_EVERY // BATCH_SIZE) == 0 and start > start_idx:
            np.savez_compressed(tmp_path, embeddings=all_embeddings, subject_ids=sid_notes)
            print(f"💾 Checkpoint saved at {tmp_path} (processed {end}/{N})")

# ---------------- Final Save ----------------
np.savez_compressed(out_path, embeddings=all_embeddings, subject_ids=sid_notes)
print(f"✅ Finished! Saved precomputed embeddings → {out_path}")

if os.path.exists(tmp_path):
    os.remove(tmp_path)
    print(f"🧹 Removed checkpoint {tmp_path}")
