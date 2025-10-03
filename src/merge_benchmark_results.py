# merge_all_metrics.py
"""
Merge all *_metrics_iter*.csv files into iteration-level and global summaries.

- Handles both per-class metric tables and AUC/Accuracy summary tables.
- Produces one CSV per iteration: results_summary_iterX.csv
- Produces a global combined CSV: results_summary_all_iterations.csv
"""

import pandas as pd
import glob
import os
from collections import defaultdict

BASE = "./"
output_global = os.path.join(BASE, "results_summary_all_iterations.csv")

# Detect all relevant *_metrics_iter*.csv files
metric_files = sorted(glob.glob(os.path.join(BASE, "*_metrics_iter*.csv")))

if not metric_files:
    print("❌ No metrics files found. Please check the directory and filenames.")
    exit(1)

# Group files by iteration (iter1, iter2, etc.)
grouped = defaultdict(list)
for file in metric_files:
    fname = os.path.basename(file)
    if "_metrics_" not in fname:
        continue
    model, suffix = fname.split("_metrics_", 1)
    run = suffix.replace(".csv", "")
    grouped[run].append((model, file))

all_summaries = []

# Process each iteration group
for run, files in grouped.items():
    dfs = []
    for model, path in files:
        df = pd.read_csv(path)

        # Normalize schema
        if list(df.columns) == ["Model", "AUC", "Accuracy"]:
            # Summary table
            df.insert(0, "iteration", run)
            df.insert(1, "model", model)
            df["Class"] = ""
            df["Precision"] = ""
            df["Recall"] = ""
            df["F1-score"] = ""
        else:
            # Per-class table
            df.insert(0, "iteration", run)
            df.insert(1, "model", model)
            if "Model" not in df.columns: df["Model"] = ""
            if "AUC" not in df.columns: df["AUC"] = ""
            if "Accuracy" not in df.columns: df["Accuracy"] = ""

        dfs.append(df)

    df_iter = pd.concat(dfs, ignore_index=True)
    iter_file = os.path.join(BASE, f"results_summary_{run}.csv")
    df_iter.to_csv(iter_file, index=False)
    print(f"✅ Saved per-iteration summary → {iter_file}")
    all_summaries.append(df_iter)

# Global merged file
df_all = pd.concat(all_summaries, ignore_index=True)
df_all.to_csv(output_global, index=False)
print(f"✅ Saved global summary → {output_global}")
