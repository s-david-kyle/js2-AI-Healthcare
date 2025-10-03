"""tfidf_logreg_notes.py
Multiclass text-only baseline: TF-IDF vectoriser + LogisticRegression
on concatenated clinical notes.
Uses ResourceLogger for cost tracking and writes metrics.
"""

import os, csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from resource_logger import ResourceLogger
import argparse

# ------------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
BASE = "./"

parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "run1")

# ------------------------------------------------------------------
# 1. Load note sequences per patient
# ------------------------------------------------------------------
notes_path = f"{BASE}/note_sequences_per_patient.npy"
notes_dict = np.load(notes_path, allow_pickle=True).item()

# Flatten notes → single doc per patient
texts = {}
for subj, adm_lists in notes_dict.items():
    all_notes = " ".join([" ".join(notes) for notes in adm_lists])
    texts[subj] = all_notes

# ------------------------------------------------------------------
# 2. Load labels (multiclass) aggregated per patient
# ------------------------------------------------------------------
feat = pd.read_csv(
    f"{BASE}/mimic_enriched_features.csv",
    usecols=["subject_id", "multiclass_label"]
)
labels = feat.groupby("subject_id")['multiclass_label'].max().astype(int)

# Drop -1 class (neither MH nor pain)
labels = labels[labels >= 0]

# Keep only patients with notes
subj_ids = sorted(set(texts.keys()) & set(labels.index))
corpus   = [texts[s] for s in subj_ids]
y        = labels.loc[subj_ids].values
subj_ids = np.array(subj_ids)

print(f"Patients with notes & valid label: {len(subj_ids)}")
print("Class distribution:", np.bincount(y))

# ------------------------------------------------------------------
# 3. Shared validation split
# ------------------------------------------------------------------
val_ids = np.load(f"shared_val_ids_{METRIC_PREFIX}.npy")
is_val = np.isin(subj_ids, val_ids)

train_idx = np.where(~is_val)[0]
val_idx   = np.where(is_val)[0]

X_train_txt = [corpus[i] for i in train_idx]
y_train     = y[train_idx]
X_test_txt  = [corpus[i] for i in val_idx]
y_test      = y[val_idx]
subj_test   = subj_ids[val_idx]

# ------------------------------------------------------------------
# 4. Vectorise + train
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(X_train_txt)
X_test  = vectorizer.transform(X_test_txt)

with ResourceLogger(tag=f"tfidf_logreg_notes_{METRIC_PREFIX}"):
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="saga",
        random_state=SEED
    )
    logreg.fit(X_train, y_train)
    prob = logreg.predict_proba(X_test)
    preds = logreg.predict(X_test)

# ------------------------------------------------------------------
# 5. Save probabilities for stacking
# ------------------------------------------------------------------
np.savez_compressed(
    f"{BASE}/tfidf_probs_{METRIC_PREFIX}.npz",
    probs=prob,
    y_true=y_test,
    subject_ids=subj_test
)

# ------------------------------------------------------------------
# 6. Metrics & save
# ------------------------------------------------------------------
accuracy = accuracy_score(y_test, preds)

# Macro-AUC (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(y_test_bin, prob, average="macro", multi_class="ovr")

report = classification_report(y_test, preds, output_dict=True, zero_division=0)

METRIC_CSV = f"{BASE}/tfidf_metrics_{METRIC_PREFIX}.csv"
with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class","Precision","Recall","F1-score"])
    for cls in ["0","1","2"]:
        row = report.get(cls, {"precision":0,"recall":0,"f1-score":0})
        writer.writerow([cls,row["precision"],row["recall"],row["f1-score"]])
    writer.writerow(["Accuracy", accuracy, "", ""])
    writer.writerow(["MacroAUC", roc_auc, "", ""])

print(f"Metrics → {METRIC_CSV}")
print("\n📊 Final TF-IDF + Logistic Regression Evaluation:")
print(f"  Macro AUC: {roc_auc:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, preds, zero_division=0, digits=4))

# ------------------------------------------------------------------
# 7. ROC Curves
# ------------------------------------------------------------------
plt.figure()
for i, cls in enumerate([0, 1, 2]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], prob[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_i:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("TF-IDF LogReg Multiclass ROC")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{BASE}/tfidf_roc_curve_{METRIC_PREFIX}.png")
plt.close()

# ------------------------------------------------------------------
# 8. Confusion Matrix
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, preds, labels=[0,1,2])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("TF-IDF LogReg Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{BASE}/tfidf_confusion_{METRIC_PREFIX}.png")
plt.close()