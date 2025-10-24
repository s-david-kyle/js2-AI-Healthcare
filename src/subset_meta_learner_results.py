#!/usr/bin/env python3
# subset_meta_learner_results.py
# ------------------------------------------------------------
# Extract only meta-learner iteration results (e.g., iterX_logisticregression)
# from results_summary_all_iterations.csv
# and save as stacker_multiclass_results.csv
# ------------------------------------------------------------

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1️⃣ Load full results file
# ------------------------------------------------------------
CSV = Path("results_summary_all_iterations.csv")
if not CSV.exists():
    raise FileNotFoundError(f"❌ Could not find {CSV.resolve()}")

df = pd.read_csv(CSV)

# ------------------------------------------------------------
# 2️⃣ Keep only rows for stacker/meta-learner iterations
# ------------------------------------------------------------
# These are models whose 'model' column = 'stacker_multiclass'
# and iteration names like "iter10_logisticregression"
df_meta = df[
    (df["model"].str.contains("stacker_multiclass", case=False, na=False))
    & (df["iteration"].str.contains("iter", case=False, na=False))
].copy()

# ------------------------------------------------------------
# 3️⃣ Keep only the relevant columns
# ------------------------------------------------------------
cols_to_keep = [
    "iteration",
    "model",
    "Class",
    "F1-score",
    "Precision",
    "Recall",
    "AUC",
]
df_meta = df_meta[cols_to_keep]

# ------------------------------------------------------------
# 4️⃣ Save clean subset
# ------------------------------------------------------------
OUT = Path("stacker_multiclass_results.csv")
df_meta.to_csv(OUT, index=False)
print(f"✅ Saved subsetted meta-learner results to: {OUT.resolve()}")
print(f"Rows retained: {len(df_meta)}")
print(df_meta.head(10))
