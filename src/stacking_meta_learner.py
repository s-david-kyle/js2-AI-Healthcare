"""
stacking_meta_learner.py
Improved stacking meta-learner that evaluates multiple classifiers and selects the best based on cross-validated macro F1 score.
"""
import os, csv, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from resource_logger import ResourceLogger
import argparse
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*")

import collections

# Reproducibility
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
rng = np.random.default_rng(SEED)


# Args and paths
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")
BASE = "."

mlflow.set_experiment("stacking_meta_learner")
mlflow.start_run(run_name=f"meta_learner_{METRIC_PREFIX}")
mlflow.log_param("seed", SEED)
mlflow.log_param("metric_prefix", METRIC_PREFIX)


# Model outputs expected in .npz format with subject_ids
model_paths = {
    "lstm":         f"{BASE}/lstm_probs_{METRIC_PREFIX}.npz",
    "gru":          f"{BASE}/gru_probs_{METRIC_PREFIX}.npz",
    "transformer":  f"{BASE}/transformer_probs_{METRIC_PREFIX}.npz",
    "clinicalbert_lstm": f"{BASE}/clinicalbert_lstm_probs_{METRIC_PREFIX}.npz",
    "rf":           f"{BASE}/rf_probs_{METRIC_PREFIX}.npz",
    "xgb":          f"{BASE}/xgb_probs_{METRIC_PREFIX}.npz",
    "tfidf":        f"{BASE}/tfidf_probs_{METRIC_PREFIX}.npz"
}

# Validate files (log only, don't exit)
missing = [m for m, path in model_paths.items() if not os.path.exists(path)]
if missing:
    log_path = f"stacking_missing_files_{METRIC_PREFIX}.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(f"{m}: {model_paths[m]}" for m in missing))
    print(f"⚠️ Missing model files logged to: {log_path}")

# Load and align predictions using subject_ids
probs_by_model, y_by_model, subj_by_model = {}, {}, {}
for name, path in model_paths.items():
    data = np.load(path)
    probs_by_model[name] = data["probs"]
    y_by_model[name] = data["y_true"]
    subj_by_model[name] = data["subject_ids"]

# Check that all models predict the same number of classes
num_class_dims = [probs.shape[1] for probs in probs_by_model.values()]
if len(set(num_class_dims)) != 1:
    print("Mismatch in number of predicted classes across models:")
    for name, probs in probs_by_model.items():
        print(f"  {name}: {probs.shape[1]} classes")
    Path(f"stacking_class_mismatch_{METRIC_PREFIX}.txt").touch()
    sys.exit(1)


# Intersect subject IDs
intersect_ids = set(int(x) for x in subj_by_model["lstm"])
for ids in subj_by_model.values():
    ids_int = np.asarray(ids).astype(int).flatten().tolist()
    intersect_ids &= set(ids_int)

if len(intersect_ids) == 0:
    print("No common subject_ids found across models.")
    Path(f"stacking_inconsistent_{METRIC_PREFIX}.txt").touch()
    sys.exit(0)

intersect_ids = sorted(intersect_ids)
X_parts = []
valid_models = []
all_feature_names = []

for name in model_paths:
    df = pd.DataFrame(probs_by_model[name])
    df["subject_id"] = subj_by_model[name]

    # Keep only intersecting IDs and ensure sort order
    df = df[df["subject_id"].isin(intersect_ids)].drop_duplicates(subset="subject_id")
    df = df.sort_values("subject_id")

    if df.shape[0] != len(intersect_ids):
        print(f"⚠️ Skipping model {name}: only has {df.shape[0]} of {len(intersect_ids)} intersect_ids")
        continue

    # Extract feature matrix and feature names
    X_model = df.drop(columns="subject_id").values
    num_features = X_model.shape[1]
    feature_names = [f"{name}_class_{i}" for i in range(num_features)]
    
    X_parts.append(X_model)
    all_feature_names.extend(feature_names)
    valid_models.append(name)

# Final stacking matrix
if len(X_parts) < 2:
    print("❌ Not enough models with matching subject_ids to stack. Exiting.")
    Path(f"stacking_incomplete_{METRIC_PREFIX}.txt").touch()
    sys.exit(1)

X_meta = np.concatenate(X_parts, axis=1)
mlflow.log_param("meta_input_dim", X_meta.shape[1])
y_true = (
    pd.DataFrame({
        "y": y_by_model["lstm"],
        "subject_id": subj_by_model["lstm"]
    }).drop_duplicates(subset="subject_id")
     .set_index("subject_id")
     .loc[intersect_ids]["y"]
     .values
)

# Determine number of unique classes
num_classes = len(np.unique(y_true))
mlflow.log_param("num_classes", num_classes)

train_idx, test_idx = train_test_split(
    np.arange(len(X_meta)), test_size=0.2, stratify=y_true, random_state=SEED
)
X_train_meta, X_test_meta = X_meta[train_idx], X_meta[test_idx]
y_train_meta, y_test_meta = y_true[train_idx], y_true[test_idx]
test_subject_ids = np.array(intersect_ids)[test_idx]

# ---------------------------------------------------------------------
# Filter out rare classes in y_train_meta that may break CV splits
# ---------------------------------------------------------------------
from collections import Counter

def filter_rare_classes(X, y, min_count=5):
    counts = Counter(y)
    valid_classes = [cls for cls, cnt in counts.items() if cnt >= min_count]
    keep_idx = np.isin(y, valid_classes)
    return X[keep_idx], y[keep_idx]

X_train_meta, y_train_meta = filter_rare_classes(X_train_meta, y_train_meta, min_count=5)
mlflow.log_param("meta_train_classes_postfilter", list(np.unique(y_train_meta)))


mlflow.log_param("meta_train_size", X_train_meta.shape[0])
mlflow.log_param("meta_test_size", X_test_meta.shape[0])

print(f"✅ Final aligned shape: {X_meta.shape} using models: {valid_models}")

# ---------------------------------------------------------------------
# 2. Evaluate candidate meta-learners
# ---------------------------------------------------------------------
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=SEED
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        class_weight="balanced", random_state=SEED,
        verbose=-1  # ✅ updated param
    ),
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=SEED),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    "NaiveBayes": GaussianNB(),
    "StackingCV": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)),
            ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric="mlogloss",
                                  verbosity=0, use_label_encoder=False,
                                  objective="multi:softprob", num_class=num_classes,
                                  random_state=SEED))
        ],
        final_estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
        cv=2
    )
}

best_model, best_score, best_fold_scores = None, -1, None
candidate_scores = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Evaluating candidate meta-learners:")

for name, model in candidates.items():
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train_meta, y_train_meta):
        model.fit(X_train_meta[train_idx], y_train_meta[train_idx])
        raw_preds = model.predict(X_train_meta[val_idx])
        preds = (np.argmax(raw_preds, axis=1)
                 if hasattr(raw_preds, "shape") and raw_preds.ndim == 2 and raw_preds.shape[1] > 1
                 else raw_preds)
        score = f1_score(y_train_meta[val_idx], preds, average="macro", zero_division=0)
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    candidate_scores[name] = fold_scores  # ✅ store per-model scores
    print(f"{name}: macro-F1 = {avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_model = (name, model)
        best_fold_scores = fold_scores  # ✅ keep the fold scores for the best model

print(f"\nBest meta-learner: {best_model[0]} (macro-F1 = {best_score:.4f})")

mlflow.log_param("best_model", best_model[0])
mlflow.log_metric("best_macro_f1", best_score)

# ---------------------------------------------------------------------
# 3. Final training and evaluation
# ---------------------------------------------------------------------
with ResourceLogger(tag=f"stacker_multiclass_{METRIC_PREFIX}"):
    best_model[1].fit(X_train_meta, y_train_meta)
    y_pred_test = best_model[1].predict(X_test_meta)

train_pred = best_model[1].predict(X_train_meta)
train_f1 = f1_score(y_train_meta, train_pred, average="macro", zero_division=0)
mlflow.log_metric("train_macro_f1", train_f1)
mlflow.log_metric("test_macro_f1", f1_score(y_test_meta, y_pred_test, average="macro", zero_division=0))

# Save model
joblib.dump(best_model[1], f"{BASE}/stacker_best_model_{METRIC_PREFIX}.pkl")
with open(f"{BASE}/stacker_best_model_{METRIC_PREFIX}.txt", "w") as f:
    f.write(best_model[0])

# Save metrics
METRIC_CSV = f"{BASE}/stacker_multiclass_metrics_{METRIC_PREFIX}_{best_model[0].lower()}.csv"
acc = accuracy_score(y_test_meta, y_pred_test)
report = classification_report(y_test_meta, y_pred_test, zero_division=0, output_dict=True)

with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score"])
    for cls in sorted(report.keys()):
        if cls not in ["accuracy", "macro avg", "weighted avg"]:
            row = report[cls]
            writer.writerow([cls, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}"])

print(f"Metrics -> {METRIC_CSV}")
print(f"\nFinal Meta-Learner ({best_model[0]}) Evaluation:")
print(classification_report(y_test_meta, y_pred_test, zero_division=0))

mlflow.sklearn.log_model(best_model[1], name="model", input_example=X_train_meta[:5])
mlflow.log_metric("accuracy", acc)
for cls, row in report.items():
    if isinstance(row, dict):
        mlflow.log_metric(f"f1_class_{cls}", row.get("f1-score", 0.0))


# Save all candidate scores
X_meta_filtered, y_true_filtered = filter_rare_classes(X_meta, y_true, min_count=5)

# ---------------------------------------------------------------------
# Save all candidate scores (avg + per-fold)
# ---------------------------------------------------------------------
score_log_path = f"{BASE}/stacker_candidate_scores_{METRIC_PREFIX}.csv"
with open(score_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Header includes fold names dynamically
    header = ["Model", "Avg_MacroF1"] + [f"Fold{i+1}" for i in range(cv.get_n_splits())]
    writer.writerow(header)

    for name, fold_scores in candidate_scores.items():
        avg_score = np.mean(fold_scores)
        row = [name, f"{avg_score:.4f}"] + [f"{s:.4f}" for s in fold_scores]
        writer.writerow(row)

print(f"📊 Candidate per-fold scores saved → {score_log_path}")
mlflow.log_artifact(score_log_path)


# ---------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(y_test_meta, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix\nMeta-Learner: {best_model[0]} ({METRIC_PREFIX})", fontsize=12)
plt.tight_layout()
plt.savefig(f"{BASE}/stacker_confusion_{METRIC_PREFIX}.png")
plt.show()

print(f"Confusion matrix saved -> stacker_confusion_{METRIC_PREFIX}.png")

mlflow.log_artifact(METRIC_CSV)
mlflow.log_artifact(score_log_path)
mlflow.log_artifact(f"{BASE}/stacker_confusion_{METRIC_PREFIX}.png")
mlflow.log_artifact(f"{BASE}/stacker_best_model_{METRIC_PREFIX}.txt")
mlflow.log_artifact(f"{BASE}/stacker_best_model_{METRIC_PREFIX}.pkl")

# ---------------------------------------------------------------------
# 5. Additional Outputs: Predictions CSV, SHAP, Per-Fold Scores
# ---------------------------------------------------------------------

# Save CSV of y_true and y_pred_test
pred_csv_path = f"{BASE}/stacker_preds_{METRIC_PREFIX}.csv"
pd.DataFrame({
    "subject_id": test_subject_ids,
    "y_true": y_test_meta,
    "y_pred": y_pred_test
}).to_csv(pred_csv_path, index=False)
print(f"📄 Predictions CSV saved → {pred_csv_path}")
mlflow.log_artifact(pred_csv_path)

# Save per-fold F1 scores (best model only)
fold_scores_log_path = f"{BASE}/stacker_best_model_folds_{METRIC_PREFIX}.csv"
with open(fold_scores_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Macro-F1"])
    for i, score in enumerate(best_fold_scores):   # ✅ use stored fold scores
        writer.writerow([f"Fold-{i+1}", f"{score:.4f}"])
mlflow.log_artifact(fold_scores_log_path)

# In model card writing:
for i, score in enumerate(best_fold_scores):       # ✅ correct scores
    f.write(f"  - Fold {i+1}: {score:.4f}\n")

mlflow.log_artifact(fold_scores_log_path)

# SHAP Analysis (for supported models only)
import shap

explainable_models = {"LightGBM", "XGBoost", "CatBoost", "LogisticRegression"}
if best_model[0] in explainable_models:
    try:
        print("🔍 Running SHAP explainability...")

        # Summarize background for efficiency
        X_background_df = pd.DataFrame(X_train_meta, columns=all_feature_names)
        X_background_sample = shap.sample(X_background_df, 100)
        explainer = shap.Explainer(best_model[1], X_background_sample)

        # Use a limited number of samples to avoid OOM
        X_explain_df = pd.DataFrame(X_train_meta[:200], columns=all_feature_names)
        shap_values = explainer(X_explain_df)

        # Save summary plot
        shap_plot_path = f"{BASE}/stacker_shap_summary_{METRIC_PREFIX}_{best_model[0].lower()}.png"
        shap.summary_plot(shap_values, X_train_meta[:200], show=False)
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        print(f"📊 SHAP summary saved → {shap_plot_path}")
        mlflow.log_artifact(shap_plot_path)

        # Save top features by mean |SHAP| value
        try:
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            top_idx = np.argsort(mean_shap)[::-1][:20]
            top_features_csv = f"{BASE}/stacker_shap_top_features_{METRIC_PREFIX}_{best_model[0].lower()}.csv"
            with open(top_features_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Feature Index", "Mean |SHAP|"])
                for i in top_idx:
                    writer.writerow([i, f"{mean_shap[i]:.6f}"])
            mlflow.log_artifact(top_features_csv)
            print(f"📄 SHAP feature importance saved → {top_features_csv}")
        except Exception as e:
            print(f"⚠️ Could not generate top feature CSV: {e}")

    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")
else:
    print(f"ℹ️ SHAP skipped: {best_model[0]} not in explainable models")

with open(f"{BASE}/stacker_meta_feature_names_{METRIC_PREFIX}.txt", "w") as f:
    for name in all_feature_names:
        f.write(name + "\n")
mlflow.log_artifact(f"{BASE}/stacker_meta_feature_names_{METRIC_PREFIX}.txt")

# ---------------------------------------------------------------------
# 6. Additional SHAP Visualizations: Force and Waterfall for a sample
# ---------------------------------------------------------------------
try:
    import shap

    if hasattr(best_model[1], "predict_proba"):
        # Choose a representative sample
        idx_sample = 0
        sample_input = X_train_meta[idx_sample:idx_sample+1]

        print("📈 Generating force and waterfall plots...")
        explainer = shap.Explainer(best_model[1], X_train_meta)
        shap_values_sample = explainer(sample_input)

        # Force plot
        force_plot_path = f"{BASE}/stacker_shap_force_{METRIC_PREFIX}.html"
        shap.save_html(force_plot_path, shap.plots.force(shap_values_sample[0], matplotlib=False))
        mlflow.log_artifact(force_plot_path)
        print(f"📌 SHAP force plot saved → {force_plot_path}")

        # Waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values_sample[0], show=False)
        waterfall_path = f"{BASE}/stacker_shap_waterfall_{METRIC_PREFIX}.png"
        plt.savefig(waterfall_path, bbox_inches="tight")
        mlflow.log_artifact(waterfall_path)
        print(f"📊 SHAP waterfall plot saved → {waterfall_path}")

except Exception as e:
    print(f"⚠️ Additional SHAP visualizations failed: {e}")

# ---------------------------------------------------------------------
# 7. Logistic Regression Coefficients Plot (if applicable)
# ---------------------------------------------------------------------
if best_model[0] == "LogisticRegression" and hasattr(best_model[1], "coef_"):
    try:
        coef = best_model[1].coef_
        n_classes, n_features = coef.shape
        fig, axes = plt.subplots(n_classes, 1, figsize=(10, 4 * n_classes))

        for i in range(n_classes):
            ax = axes[i] if n_classes > 1 else axes
            ax.bar(range(n_features), coef[i])
            ax.set_title(f"LogisticRegression Coefficients - Class {i}")
            ax.set_xlabel("Meta-Feature Index")
            ax.set_ylabel("Weight")

        coef_plot_path = f"{BASE}/logreg_coef_{METRIC_PREFIX}.png"
        plt.tight_layout()
        plt.savefig(coef_plot_path)
        mlflow.log_artifact(coef_plot_path)
        print(f"📉 Coefficient plot saved → {coef_plot_path}")
    except Exception as e:
        print(f"⚠️ Coefficient plot failed: {e}")

# ---------------------------------------------------------------------
# 8. Model Card-style Interpretability Summary
# ---------------------------------------------------------------------
model_card_path = f"{BASE}/model_card_{METRIC_PREFIX}.md"
with open(model_card_path, "w") as f:
    f.write(f"# Model Card: Meta-Learner ({best_model[0]})\n\n")
    f.write(f"**Metric Prefix**: `{METRIC_PREFIX}`\n")
    f.write(f"**Selected Model**: `{best_model[0]}`\n\n")

    f.write("## Performance Summary\n")
    f.write(f"- Accuracy: {acc:.4f}\n")
    f.write(f"- Macro F1-score: {best_score:.4f}\n")
    f.write("- Per-fold F1 Scores (Best Model):\n")
    for i, score in enumerate(best_fold_scores):
        f.write(f"  - Fold {i+1}: {score:.4f}\n")
    f.write("\n")

    f.write("## Candidate Comparison (Avg Macro-F1)\n")
    for name, fold_scores in candidate_scores.items():
        f.write(f"- {name}: {np.mean(fold_scores):.4f}\n")


    f.write("## Interpretability Artifacts\n")
    if os.path.exists(shap_plot_path):
        f.write(f"- SHAP Summary Plot: `{shap_plot_path}`\n")
    if best_model[0] == "LogisticRegression" and os.path.exists(coef_plot_path):
        f.write(f"- Logistic Regression Coefficients Plot: `{coef_plot_path}`\n")
    if os.path.exists(force_plot_path):
        f.write(f"- SHAP Force Plot: `{force_plot_path}`\n")
    if os.path.exists(waterfall_path):
        f.write(f"- SHAP Waterfall Plot: `{waterfall_path}`\n")
    
    f.write("\n## Notes\n")
    f.write("- This model stacks multiple base learners using a meta-classifier trained on probability outputs.\n")
    f.write("- All training and evaluation used shared validation splits across all models.\n")
    f.write("- SHAP explanations and coefficients (if available) provide insight into which base models contributed most to predictions.\n")

print(f"📄 Model card saved → {model_card_path}")
mlflow.log_artifact(model_card_path)

# Finally 
mlflow.end_run()
