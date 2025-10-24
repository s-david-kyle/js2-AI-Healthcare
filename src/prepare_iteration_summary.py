#!/usr/bin/env python3
"""
prepare_iteration_summary.py
----------------------------------------
Subset results_summary_all_iterations.csv to latest 30 iterations
and produce aggregated statistics (F1, Precision, Recall) for plotting.
----------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

CSV = Path("../analysis/results/results_summary_all_iterations.csv")
OUT_DIR = Path("../analysis/results/results_clean")
OUT_DIR.mkdir(exist_ok=True)

print("📂 Loading results_summary_all_iterations.csv...")
df = pd.read_csv(CSV, on_bad_lines="skip", low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ---------------------------------------------------------------------
# 1. Detect iteration column
# ---------------------------------------------------------------------
iteration_col = None
for c in ["iter", "iteration", "run"]:
    if c in df.columns:
        iteration_col = c
        break

if iteration_col is None:
    raise ValueError("❌ No iteration column found in results_summary_all_iterations.csv")

# Extract numeric iteration number
df["iter_num"] = df[iteration_col].astype(str).str.extract(r"(\d+)").astype(float)

# ---------------------------------------------------------------------
# 2. Keep only latest 30 iterations
# ---------------------------------------------------------------------
max_iter = int(df["iter_num"].max())
keep_iters = list(range(max_iter - 29, max_iter + 1))
df_subset = df[df["iter_num"].isin(keep_iters)].copy()

print(f"✅ Keeping last 30 iterations (iter{keep_iters[0]}–iter{keep_iters[-1]})")
print(f"   → {len(df_subset)} rows retained")

# ---------------------------------------------------------------------
# 3. Normalize model names (handles duplicates, ensures consistency)
# ---------------------------------------------------------------------
model_candidates = [c for c in df_subset.columns if c.lower() == "model"]
if not model_candidates:
    raise ValueError("❌ No model-like column found in results_summary_all_iterations.csv")

if len(model_candidates) > 1:
    print(f"⚠️ Multiple model columns found: {model_candidates}. Merging them.")
    df_subset["model_merged"] = df_subset[model_candidates].bfill(axis=1).iloc[:, 0]
    df_subset.drop(columns=model_candidates, inplace=True, errors="ignore")
    df_subset.rename(columns={"model_merged": "model"}, inplace=True)
else:
    df_subset.rename(columns={model_candidates[0]: "model"}, inplace=True)

df_subset["model"] = df_subset["model"].astype(str)

# Clean model names
df_subset["model"] = (
    df_subset["model"]
    .str.lower()
    .replace({
        "stacker_": "stack_",
        "clinicalbert_transformer": "Clinicalbert",
        "_multiclass": "",
        "gru": "GRU",
        "lstm": "LSTM",
        "tfidf": "TF-IDF",
        "transformer": "Transformer",
        "tabular_xgb": "XGBoost",
        "tabular_rf": "Random Forest",
    }, regex=True)
    .str.replace("Clinicalbert", "ClinicalBERT ", regex=False)
    .str.replace("stack_", "Stacked ", regex=False)
    .str.strip()
)

print("✅ Model column successfully normalized.")
print("   Unique models detected:", df_subset["model"].nunique())

# ---------------------------------------------------------------------
# 4. Convert relevant metric columns
# ---------------------------------------------------------------------
metric_cols = [c for c in df_subset.columns if any(m in c for m in ["f1", "precision", "recall", "accuracy", "auc"])]
for col in metric_cols:
    df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

# ---------------------------------------------------------------------
# 5. Compute macro-averaged metrics per iteration
# ---------------------------------------------------------------------
print("📊 Computing macro-averaged metrics per iteration...")

# Keep only numeric classes
df_subset = df_subset[pd.to_numeric(df_subset["class"], errors="coerce").notnull()]
df_subset["class"] = df_subset["class"].astype(int)

# Group by iteration + model to compute macro metrics
macro_iter = (
    df_subset.groupby(["model", "iteration"], as_index=False)
    .agg({
        "f1-score": "mean",
        "precision": "mean",
        "recall": "mean"
    })
    .rename(columns={
        "f1-score": "macro_f1",
        "precision": "macro_precision",
        "recall": "macro_recall"
    })
)

# ---------------------------------------------------------------------
# 6. Compute mean ± std across iterations
# ---------------------------------------------------------------------
agg_summary = (
    macro_iter.groupby("model", as_index=False)
    .agg({
        "macro_f1": ["mean", "std"],
        "macro_precision": ["mean", "std"],
        "macro_recall": ["mean", "std"]
    })
)

# Flatten column names
agg_summary.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in agg_summary.columns.values
]

# Sort by mean macro F1
agg_summary = agg_summary.sort_values("macro_f1_mean", ascending=False)

# ---------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------
agg_path = OUT_DIR / "results_summary_agg.csv"
macro_iter_path = OUT_DIR / "results_summary_macro_iter.csv"

macro_iter.to_csv(macro_iter_path, index=False)
agg_summary.to_csv(agg_path, index=False)

print(f"💾 Saved macro per-iteration → {macro_iter_path}")
print(f"💾 Saved aggregated summary → {agg_path}")

# ---------------------------------------------------------------------
# 8. Preview
# ---------------------------------------------------------------------
print("\n📊 Top Models by Mean Macro F1:")
print(agg_summary[["model", "macro_f1_mean", "macro_f1_std", "macro_precision_mean", "macro_recall_mean"]].head(10).to_string(index=False))
