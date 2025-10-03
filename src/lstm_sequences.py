import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from collections import Counter

# ---------------------------------------------------------------------
# Set reproducible but flexible seed
# ---------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)

METRIC_PREFIX = os.getenv("METRIC_PREFIX", "iter1")

# ---------------------------------------------------------------------
# 1. Load boosted structured + note features
# ---------------------------------------------------------------------
print("🔹 Loading data...")
df = pd.read_csv("mimic_enriched_features_w_notes.csv")

# ---------------------------------------------------------------------
# 2. Feature Selection
# ---------------------------------------------------------------------
feature_cols = [
    "approx_age", "gender", "insurance_group", "admission_type", "length_of_stay",
    "was_in_icu", "seen_by_psych", "polypharmacy_flag", "psych_or_pain_rx_count",
    "note_count", "avg_note_length", "sentiment", "note_cluster",
] + [f"tfidf_{term}" for term in [
    'pain', 'anxiety', 'depression', 'headache', 'fatigue', 'sleep',
    'sad', 'crying', 'hopeless', 'tired', 'insomnia', 'nausea', 'vomiting'
]] + [
    f"topic_{i+1}" for i in range(5)
] + [
    "pca_1", "pca_2", "umap_1", "umap_2",
    "transfer_count"
]

label_col = "multiclass_label"

# ---------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------
print("🔹 Preprocessing...")

df[feature_cols] = df[feature_cols].fillna(0)
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

df["gender"] = df["gender"].map({"M": 1, "F": 0})
df["insurance_group"] = df["insurance_group"].astype("category").cat.codes
df["admission_type"] = df["admission_type"].astype("category").cat.codes

for col in ["was_in_icu", "seen_by_psych", "polypharmacy_flag"]:
    df[col] = df[col].astype(int)

# ---------------------------------------------------------------------
# 4. Build visit-level sequences (with subject IDs)
# ---------------------------------------------------------------------
print("🔹 Building sequences...")

SEQUENCE_LENGTH = 10
sequences, labels, subject_ids = [], [], []

for subject_id, group in df.sort_values("admittime").groupby("subject_id"):
    visit_features = group[feature_cols].values
    label_sequence = group[label_col].values

    if len(visit_features) == 0:
        continue

    if len(visit_features) >= SEQUENCE_LENGTH:
        visit_features = visit_features[-SEQUENCE_LENGTH:]
        label = label_sequence[-1]
    else:
        pad_len = SEQUENCE_LENGTH - len(visit_features)
        visit_features = np.pad(
            visit_features, ((pad_len, 0), (0, 0)), mode="constant"
        )
        label = label_sequence[-1]

    sequences.append(visit_features)
    labels.append(label)
    subject_ids.append(subject_id)

X = np.stack(sequences)
y = np.array(labels)
subject_ids = np.array(subject_ids)

# ---------------------------------------------------------------------
# 5. Shared validation split
# ---------------------------------------------------------------------
print("🔹 Splitting train/val with shared IDs...")
val_ids = np.load(f"shared_val_ids_{METRIC_PREFIX}.npy")

is_val = np.isin(subject_ids, val_ids)
X_train, X_val = X[~is_val], X[is_val]
y_train, y_val = y[~is_val], y[is_val]
subj_train, subj_val = subject_ids[~is_val], subject_ids[is_val]

# Diagnostic: check class coverage
def check_class_coverage(y, label):
    classes, counts = np.unique(y, return_counts=True)
    print(f"📦 {label}:", dict(zip(classes, counts)))
    missing = set([0, 1, 2]) - set(classes)
    if missing:
        print(f"⚠️ WARNING: {label} missing classes: {missing}")
    return missing

check_class_coverage(y_train, "Train")
check_class_coverage(y_val, "Validation")

# Optional oversampling of co-morbid (class 2)
OVERSAMPLE_TARGET = int(os.getenv("OVERSAMPLE_TARGET", 100))
train_counts = Counter(y_train)
minority_class = 2
if train_counts[minority_class] < OVERSAMPLE_TARGET:
    print("⚠️ Oversampling class 2 (co-morbid)...")
    X_min = X_train[y_train == minority_class]
    y_min = y_train[y_train == minority_class]
    n_samples = OVERSAMPLE_TARGET - len(y_min)

    X_upsampled, y_upsampled = resample(
        X_min, y_min, replace=True, n_samples=n_samples, random_state=SEED
    )
    X_train = np.concatenate([X_train, X_upsampled])
    y_train = np.concatenate([y_train, y_upsampled])
    print("✅ Oversampling complete.")
    check_class_coverage(y_train, "Train (post-oversample)")

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
print("🔹 Saving sequences...")

out_dir = "./"
os.makedirs(out_dir, exist_ok=True)
np.save(f"{out_dir}/X_train_seq.npy", X_train)
np.save(f"{out_dir}/y_train_seq.npy", y_train)
np.save(f"{out_dir}/X_val_seq.npy", X_val)
np.save(f"{out_dir}/y_val_seq.npy", y_val)
np.save(f"{out_dir}/subject_ids_train_seq.npy", subj_train)
np.save(f"{out_dir}/subject_ids_val_seq.npy", subj_val)

# Save full set
np.save(f"{out_dir}/X_seq.npy", X)
np.save(f"{out_dir}/y_seq.npy", y)
np.save(f"{out_dir}/subject_ids_seq.npy", subject_ids)

# Save feature columns
with open(f"{out_dir}/feature_cols.txt", "w") as f:
    for col in feature_cols:
        f.write(f"{col}\n")

print("✅ Saved sequence arrays aligned with shared validation IDs.")

# ---------------------------------------------------------------------
# 7. Summary Report
# ---------------------------------------------------------------------
print("\n📋 Summary Report:")
print(f"Training Set: {X_train.shape}, Validation Set: {X_val.shape}")
print(f"Feature Dimensions: {X_train.shape[-1]}")
print(f"Class Distribution (Train):", dict(Counter(y_train)))
print(f"Class Distribution (Validation):", dict(Counter(y_val)))
print(f"Seed Used: {SEED}, Metric Prefix: {METRIC_PREFIX}")
