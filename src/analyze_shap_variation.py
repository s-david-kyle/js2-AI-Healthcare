#!/usr/bin/env python3
# analyze_shap_variation.py
# ---------------------------------------------------------------------
# Automatically detects SHAP model-importance files (any folder depth),
# aggregates them across iterations, and visualizes variation.
# ---------------------------------------------------------------------

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Auto-locate the most likely results directory
# ---------------------------------------------------------------------
CANDIDATES = ["./", "./results", "./output", "./outputs", "./artifacts"]
BASE = None

for cand in CANDIDATES:
    found = glob.glob(os.path.join(cand, "stacker_shap_model_importance*.csv"))
    if found:
        BASE = cand
        break

if BASE is None:
    raise FileNotFoundError("❌ No SHAP model importance CSV files found in any expected directory.")

print(f"📂 Using SHAP results from: {Path(BASE).resolve()}")

OUT_DIR = Path(BASE) / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Locate all SHAP importance CSV files (StackingCV + standard)
# ---------------------------------------------------------------------
files = sorted(
    glob.glob(f"{BASE}/stacker_shap_model_importance_*_stackingcv_base_importance.csv")
    + glob.glob(f"{BASE}/stacker_shap_model_importance_iter*_*.csv")
)

if not files:
    raise FileNotFoundError("❌ No SHAP model importance CSV files found after search.")

print(f"🔍 Found {len(files)} SHAP importance files:")
for f in files:
    print("  -", f)

# ---------------------------------------------------------------------
# 3. Parse iteration IDs and combine into one dataframe
# ---------------------------------------------------------------------
dfs = []
for fpath in files:
    df = pd.read_csv(fpath)
    basename = os.path.basename(fpath)
    iter_tag = "unknown"
    for part in basename.split("_"):
        if part.startswith("iter"):
            iter_tag = part
            break

    df["iteration"] = iter_tag

    # Normalize column names
    if "BaseModel" in df.columns:
        df = df.rename(columns={"BaseModel": "model", "MeanAbsSHAP": "mean_abs_shap"})
    elif "model" not in df.columns:
        raise ValueError(f"Unrecognized column format in {basename}")

    dfs.append(df[["model", "mean_abs_shap", "iteration"]])

all_df = pd.concat(dfs, ignore_index=True)
all_df["mean_abs_shap"] = pd.to_numeric(all_df["mean_abs_shap"], errors="coerce")

# ---------------------------------------------------------------------
# 4. Aggregate statistics per model
# ---------------------------------------------------------------------
summary = (
    all_df.groupby("model")["mean_abs_shap"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

summary_path = OUT_DIR / "shap_importance_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"📄 Summary saved → {summary_path}")

# ---------------------------------------------------------------------
# 5. Boxplot across iterations
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_df, x="model", y="mean_abs_shap", palette="viridis", showfliers=False)
sns.stripplot(data=all_df, x="model", y="mean_abs_shap", color="black", alpha=0.4, jitter=True)
plt.title("SHAP Importance Variation Across Iterations")
plt.xlabel("Base Model")
plt.ylabel("Mean |SHAP| Value")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
boxplot_path = OUT_DIR / "shap_variation_boxplot.png"
plt.savefig(boxplot_path, dpi=300)
plt.close()
print(f"📊 Boxplot saved → {boxplot_path}")

# ---------------------------------------------------------------------
# 6. Heatmap of model × iteration variation
# ---------------------------------------------------------------------
pivot = all_df.pivot_table(index="iteration", columns="model", values="mean_abs_shap", aggfunc="mean")
plt.figure(figsize=(10, max(4, 0.5 * len(pivot))))
sns.heatmap(pivot, cmap="mako", annot=True, fmt=".3f")
plt.title("Mean |SHAP| per Base Model Across Iterations")
plt.xlabel("Base Model")
plt.ylabel("Iteration")
plt.tight_layout()
heatmap_path = OUT_DIR / "shap_variation_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"📊 Heatmap saved → {heatmap_path}")

# ---------------------------------------------------------------------
# 7. Line plot (stability trajectories)
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_df, x="iteration", y="mean_abs_shap", hue="model", marker="o")
plt.title("SHAP Importance Stability Across Iterations")
plt.xlabel("Iteration")
plt.ylabel("Mean |SHAP| Value")
plt.legend(title="Base Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
lineplot_path = OUT_DIR / "shap_variation_lineplot.png"
plt.savefig(lineplot_path, dpi=300)
plt.close()
print(f"📈 Line plot saved → {lineplot_path}")

# ---------------------------------------------------------------------
# 8. Save supporting pivot table for additional stats
# ---------------------------------------------------------------------
pivot.to_csv(OUT_DIR / "shap_importance_pivot.csv")
print(f"📄 Pivot table saved → {OUT_DIR / 'shap_importance_pivot.csv'}")

# ---------------------------------------------------------------------
# 9. Summary table preview
# ---------------------------------------------------------------------
print("\n📊 SHAP Model Importance Stability Summary:")
print(summary.round(4).to_string(index=False))
