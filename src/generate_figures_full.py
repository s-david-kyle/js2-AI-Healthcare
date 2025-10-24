#!/usr/bin/env python3
# ============================================================
# Jetstream2 Fellowship - Clean Static Figures (Python only)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# Setup & Theme
# ---------------------------------------------------------------------
OUT = Path("plots_clean")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.1)
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

print("🎨 Generating static, publication-ready figures...")

# Helper to sanitize columns
def clean_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# FIGURE 1 — Macro F1-score Across Models
# ============================================================
# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------
CSV = Path("../analysis/results/results_clean/results_summary_agg.csv")
df = pd.read_csv(CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Identify F1 mean/std columns
mean_col = next((c for c in df.columns if "f1" in c and "mean" in c), None)
std_col  = next((c for c in df.columns if "f1" in c and "std"  in c), None)
if not mean_col or not std_col:
    raise ValueError(f"❌ Could not find F1 mean/std columns in {CSV}")
print(f"✅ Using columns: {mean_col}, {std_col}")

# ----------------------------------------------------------
# 2. Normalize model names consistently
# ----------------------------------------------------------
rename_map = {
    # --- Classical & Transformer Models ---
    "tf-idf": "TF-IDF",
    "tfidf": "TF-IDF",
    "transformer": "Transformer",
    "clinicalbert": "ClinicalBERT",
    "clinicalbert_transformer": "ClinicalBERT",
    "bert_transformer": "ClinicalBERT",

    # --- Sequence Models ---
    "gru": "GRU",
    "lstm": "LSTM",

    # --- Tabular Models ---
    "xgboost": "XGBoost",
    "tabular_xgb": "XGBoost",
    "rf": "Random Forest",
    "tabular_rf": "Random Forest",

    # --- Meta-Learners ---
    "stacked logisticregression": "Meta-Learner (Log Regn)",
    "stacked stackingcv": "Meta-Learner (Stacking CV)",
    "stacked svm": "Meta-Learner (SVM)",
}

# Lowercase and map
df["model"] = (
    df["model"]
    .astype(str)
    .str.lower()
    .str.strip()
    .replace(rename_map, regex=False)
)

# Final title-case correction for display consistency
df["model"] = df["model"].replace({"random forest": "Random Forest"}, regex=False)

print("✅ Model names standardized:")
print(df["model"].unique())

# Drop missing or unmapped models
df = df[df["model"].notna() & (df["model"] != "nan")]

# ----------------------------------------------------------
# 3. Sort by performance
# ----------------------------------------------------------
df = df.sort_values(mean_col, ascending=False)

# ----------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(8, 6))

plt.errorbar(
    df[mean_col],
    df["model"].astype(str),
    xerr=df[std_col],
    fmt="o",
    ecolor="gray",
    elinewidth=1.2,
    capsize=3,
    markersize=7,
    color="#1f77b4"
)

plt.xlabel("Macro F1-score (Mean ± SD)")
plt.ylabel("")
plt.xlim(0, 1)
#plt.title("Figure 1. Macro F1-score Across Models", pad=15)
plt.tight_layout()

# ----------------------------------------------------------
# 5. Save
# ----------------------------------------------------------
out_dir = Path("plots_clean")
out_dir.mkdir(exist_ok=True)
outfile = out_dir / "figure1_f1_dot_error_clean.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"🎨 Clean Figure 1 saved → {outfile}")


# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------
CSV = Path("../analysis/results/resource_usage.csv")
df = pd.read_csv(CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

required = {"tag", "elapsed_hr", "gpu_hrs", "cpu_pct", "disk_used_gb"}
if not required.issubset(df.columns):
    raise ValueError(f"❌ Missing required columns. Found: {list(df.columns)}")

# ----------------------------------------------------------
# 2. Normalize model names
# ----------------------------------------------------------
def clean_model(tag):
    tag = str(tag).lower()
    if "tabular_rf_xgb" in tag:
        return "RF+XGB"
    elif "tabular_logreg" in tag:
        return "Logistic Regression"
    elif "clinicalbert" in tag:
        return "ClinicalBERT"
    elif "transformer" in tag:
        return "Transformer"
    elif "gru" in tag:
        return "GRU"
    elif "lstm" in tag:
        return "LSTM"
    elif "tfidf" in tag:
        return "TF-IDF"
    elif "stacker_multiclass" in tag:
        return "Meta-Learner"
    else:
        return "Other"

df["model_type"] = df["tag"].apply(clean_model)

# Extract iteration number
df["iter_num"] = df["tag"].str.extract(r"run(\d+)").fillna(0).astype(int)

# ----------------------------------------------------------
# 3. Keep only last 30 per model type
# ----------------------------------------------------------
df = df.sort_values(["model_type", "iter_num"])
df_latest = df.groupby("model_type", group_keys=False).apply(lambda g: g.tail(30))

# ----------------------------------------------------------
# 4. Aggregate by model type
# ----------------------------------------------------------
agg = (
    df_latest.groupby("model_type")[["cpu_pct", "gpu_hrs", "disk_used_gb", "elapsed_hr"]]
    .mean()
    .reset_index()
)

# Sort by GPU hours for visual order
agg = agg.sort_values("gpu_hrs", ascending=False)

# ----------------------------------------------------------
# 5. Plot – cleaner horizontal bars
# ----------------------------------------------------------
sns.set(style="whitegrid", context="talk", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# CPU %
sns.barplot(data=agg, y="model_type", x="cpu_pct", ax=axes[0], color="#4C72B0")
axes[0].set_title("CPU Utilization (%)")
axes[0].set_xlabel("Average CPU %")
axes[0].set_ylabel("")

# GPU hours
sns.barplot(data=agg, y="model_type", x="gpu_hrs", ax=axes[1], color="#55A868")
axes[1].set_title("GPU Usage (Hours)")
axes[1].set_xlabel("Average GPU Hours")
axes[1].set_ylabel("")

for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)

plt.tight_layout()
out_dir = Path("plots_clean")
out_dir.mkdir(exist_ok=True)
outfile = out_dir / "figure2_resource_usage_by_model.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"🎨 Figure 2 saved → {outfile}")

# ============================================================
# Figure 3 — Wilcoxon Signed-Rank Heatmap (Significance Annotated)
# ============================================================

CSV = Path("../analysis/results/wilcoxon_results.csv")
df = pd.read_csv(CSV)
df.columns = [c.strip().lower() for c in df.columns]

# Detect p-value column
pcol = "p_value_adj" if "p_value_adj" in df.columns else "p_value"

# --- Clean & coerce -----------------------------------------------------------
df = (
    df[["model_1", "model_2", pcol]]
    .dropna()
    .assign(**{pcol: pd.to_numeric(df[pcol], errors="coerce")})
    .dropna()
)

# Combine duplicates conservatively (keep smallest adjusted p-value)
df = df.groupby(["model_1", "model_2"], as_index=False)[pcol].min()

# All model names
models = sorted(set(df["model_1"]) | set(df["model_2"]))

# --- Symmetric matrix ---------------------------------------------------------
mat = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
for _, r in df.iterrows():
    m1, m2, p = r["model_1"], r["model_2"], r[pcol]
    mat.loc[m1, m2] = p
    mat.loc[m2, m1] = p

# --- Rename for consistent labels across figures ------------------------------
rename_map = {
    # --- Classical & Transformer Models ---
    "tf-idf": "TF-IDF",
    "tfidf": "TF-IDF",
    "transformer": "Transformer",
    "clinicalbert": "ClinicalBERT",
    "clinicalbert_transformer": "ClinicalBERT",
    "bert_transformer": "ClinicalBERT",

    # --- Sequence Models ---
    "gru": "GRU",
    "lstm": "LSTM",

    # --- Tabular Models ---
    "xgboost": "XGBoost",
    "tabular_xgb": "XGBoost",
    "rf": "Random Forest",
    "tabular_rf": "Random Forest",

    # --- Meta-Learners ---
    "Stacked logisticregression": "Meta-Learner\n(Log Regn)",
    "stacker_logisticregression": "Meta-Learner\n(Log Regn)",
    "Stacked stackingcv": "Meta-Learner\n(Stacking CV)",
    "stacker_stackingcv": "Meta-Learner\n(Stacking CV)",
    "Stacked svm": "Meta-Learner\n(SVM)",
    "stacker_svm": "Meta-Learner\n(SVM)",
}

# Apply renaming to both index and columns
mat.rename(index=rename_map, columns=rename_map, inplace=True)

# --- Final cleanup ------------------------------------------------------------
mat = mat.apply(pd.to_numeric, errors="coerce").fillna(1.0)
np.fill_diagonal(mat.values, 1.0)

print("✅ Model names standardized:")
print(list(mat.columns))

# --- Build annotation mask ----------------------------------------------------
annot_text = mat.copy()
# --- Build annotation mask ----------------------------------------------------
def format_sig(x):
    """Return significance stars based on adjusted p-value."""
    if x < 0.001:
        return "***"
    elif x < 0.01:
        return "**"
    elif x < 0.05:
        return "*"
    else:
        return f"{x:.2f}"

annot_text = mat.copy().applymap(format_sig)

# --- Plot ---------------------------------------------------------------------
sns.set_theme(style="white", context="talk", font_scale=0.8)
fig, ax = plt.subplots(figsize=(8, 8))

# Mask upper triangle to avoid duplicate entries
mask = np.triu(np.ones_like(mat, dtype=bool))

# Draw heatmap
sns.heatmap(
    mat,
    mask=mask,
    cmap="coolwarm_r",
    vmin=0, vmax=1,
    annot=annot_text,
    fmt="",
    annot_kws={"size": 8, "weight": "bold"},
    linewidths=0.5,
    cbar=True,
    ax=ax,
)

# --- Style --------------------------------------------------------------------
#ax.set_title("Wilcoxon Signed-Rank Tests (Adjusted p-values)", pad=40)
ax.set_xlabel("")
ax.set_ylabel("")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# --- Move and format colorbar -------------------------------------------------
# Grab the colorbar object
cbar = ax.collections[0].colorbar

# Reposition to top, horizontal
cbar.ax.set_position([0.25, 0.92, 0.5, 0.03])  # [left, bottom, width, height]
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Holm-adjusted p-value", fontsize=14)

# --- Caption ------------------------------------------------------------------
caption_text = "* denotes Hold-adjusted p-values < 0.05; ** < 0.01; *** < 0.001"
fig.text(0.5, 0.05, caption_text, ha="center", fontsize=9, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 0.9])
#plt.show()

# --- Save ---------------------------------------------------------------------
out_dir = Path("plots_clean"); out_dir.mkdir(exist_ok=True)
outfile = out_dir / "figure3_wilcoxon_clean_sigmask_caption.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"🎨 Figure 3 saved → {outfile}")

# ============================================================
# FIGURE 4 — Meta-Learner Performance (± SD)
# ============================================================
try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    OUT = Path("../analysis/results/results_clean/plots_clean")
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Load & clean ---------------------------------------------------------
    df_meta = pd.read_csv("../analysis/results/results_clean/meta_learner_results.csv")

    # Normalize column names
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]

    # Expected columns
    type_col = "meta_type"
    mean_col = "mean_f1"
    sd_col = "std_f1"

    # Ensure numeric
    df_meta[mean_col] = pd.to_numeric(df_meta[mean_col], errors="coerce")
    df_meta[sd_col] = pd.to_numeric(df_meta[sd_col], errors="coerce")

    # --- Aggregate by meta_type ----------------------------------------------
    df_summary = (
        df_meta.groupby(type_col, as_index=False)
        .agg(
            mean_f1_mean=(mean_col, "mean"),
            mean_f1_sd=(mean_col, "std")  # variation across runs
        )
    )

    # Clean labels for display
    rename_map = {
        "logisticregression": "Logistic\nRegression",
        "stackingcv": "Stacking CV",
        "svm": "SVM",
        "mlp": "MLP",
        "naivebayes": "Naive Bayes",
        "randomforest": "Random Forest",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "stacker": "Stacker",
    }
    df_summary[type_col] = df_summary[type_col].astype(str).str.lower().replace(rename_map)

    # Sort by mean F1
    df_summary = df_summary.sort_values("mean_f1_mean", ascending=False)

    # --- Plot -----------------------------------------------------------------
    plt.figure(figsize=(6, 4.5))
    ax = sns.barplot(
        data=df_summary,
        y=type_col,
        x="mean_f1_mean",
        palette="mako",
        alpha=0.9,
        errorbar=None,
    )

    # Add manual error bars (horizontal)
    for i, row in enumerate(df_summary.itertuples()):
        ax.errorbar(
            x=row.mean_f1_mean,
            y=i,
            xerr=row.mean_f1_sd,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=0,
            zorder=5,
        )

    # Add numeric labels just past the end of the error bar
    for i, row in enumerate(df_summary.itertuples()):
        label_x = row.mean_f1_mean + row.mean_f1_sd + 0.015
        ax.text(
            label_x,
            i,
            f"{row.mean_f1_mean:.3f}",
            ha="left",
            va="center",
            fontsize=10
        )

    # --- Style ----------------------------------------------------------------
    ax.axhline(
        y=2.5,             # halfway between indices 5 (MLP) and 6 (SVM)
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        zorder=0,
    )
    ax.set_xlabel("Mean Macro F1 ± SD", labelpad=8)
    ax.set_ylabel("")
    ax.set_xlim(0, 0.65)
    sns.despine()
    plt.tight_layout()

    # --- Save -----------------------------------------------------------------
    out_file = OUT / "figure4_meta_learner_clean_rotated.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Figure 4 saved → {out_file}")

except Exception as e:
    print(f"⚠️ Figure 4 failed: {e}")