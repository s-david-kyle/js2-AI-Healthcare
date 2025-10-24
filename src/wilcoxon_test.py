# wilcoxon_test.py
import pandas as pd
from scipy.stats import wilcoxon
import glob, sys, itertools
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------
# 1. Load all iteration summaries
# ---------------------------------------------------------------------
files = glob.glob("../analysis/results/results_clean/results_summary_macro_iter*.csv")
if not files:
    print("❌ No results_summary_iter*.csv files found.")
    sys.exit(1)

dfs = []
for f in files:
    try:
        df_tmp = pd.read_csv(f)
        dfs.append(df_tmp)
    except Exception as e:
        print(f"⚠️ Skipping {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"📂 Loaded {len(files)} iteration summaries ({len(df)} rows)")

# ---------------------------------------------------------------------
# 2. Normalize + deduplicate columns
# ---------------------------------------------------------------------
df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

# Deduplicate column names (keep first occurrence)
df = df.loc[:, ~df.columns.duplicated()]

# Ensure we only have ONE "model" column
model_cols = [c for c in df.columns if "model" in c]
if not model_cols:
    print(f"❌ No model column found. Columns: {df.columns.tolist()}")
    sys.exit(1)

if len(model_cols) > 1:
    print(f"⚠️ Multiple model columns after deduplication: {model_cols}. Using '{model_cols[0]}'")

model_col = model_cols[0]

# ---------------------------------------------------------------------
# 3. Pick F1 column
# ---------------------------------------------------------------------
possible_f1_cols = ["macro_f1", "f1_macro", "f1_score", "f1"]
metric_col = None
for col in possible_f1_cols:
    if col in df.columns:
        metric_col = col
        break

if metric_col is None:
    print(f"❌ Could not find any F1 metric column. Available columns: {df.columns.tolist()}")
    sys.exit(1)

print(f"✅ Using F1 column: {metric_col}")

# ---------------------------------------------------------------------
# 4. Ensure iteration column
# ---------------------------------------------------------------------
if "iter" not in df.columns and "iteration" in df.columns:
    df = df.rename(columns={"iteration": "iter"})

if "iter" not in df.columns:
    print("❌ No 'iter' column available.")
    sys.exit(1)

# ---------------------------------------------------------------------
# 5. Pivot to iter × model
# ---------------------------------------------------------------------
pivot = df.pivot_table(
    index="iter", columns=model_col, values=metric_col, aggfunc="first"
)

print(f"📊 Pivot shape: {pivot.shape}, models = {list(pivot.columns)}")

# ---------------------------------------------------------------------
# 6. Pairwise Wilcoxon Tests
# ---------------------------------------------------------------------
results = []
models = list(pivot.columns)

for m1, m2 in itertools.combinations(models, 2):
    sub = pivot.dropna(subset=[m1, m2])
    if len(sub) < 5:
        continue
    stat, p = wilcoxon(sub[m1], sub[m2], alternative="two-sided")
    results.append({
        "model_1": m1,
        "model_2": m2,
        "n_pairs": len(sub),
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05
    })

if not results:
    print("⚠️ No valid model pairs found for Wilcoxon testing.")
    sys.exit(0)

# After building results_df
results_df = pd.DataFrame(results).sort_values("p_value")

# Apply Holm–Bonferroni correction
reject, p_adj, _, _ = multipletests(results_df["p_value"], method="holm")

results_df["p_value_adj"] = p_adj
results_df["significant_adj"] = reject

print("\n📊 Pairwise Wilcoxon Signed-Rank Tests (Holm-adjusted)")
print("----------------------------------------------------------")
print(results_df[["model_1", "model_2", "n_pairs", "p_value", "p_value_adj", "significant_adj"]]
      .to_string(index=False))

# Save results
results_df.to_csv("../analysis/results/wilcoxon_results.csv", index=False)
print("\n✅ Saved adjusted results → wilcoxon_results.csv")