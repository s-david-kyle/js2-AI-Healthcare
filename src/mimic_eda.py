# mimic_eda.ipynb or mimic_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
df = pd.read_csv("mimic_enriched_features.csv")
print(f"✅ Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# ---------------------------------------------------------------------
# 2. Basic overview
# ---------------------------------------------------------------------
print("\n📊 Column info:")
print(df.info())

print("\n🔍 Class distribution:")
print(df["multiclass_label"].value_counts(dropna=False).sort_index())

# ---------------------------------------------------------------------
# 3. Visualize distributions
# ---------------------------------------------------------------------
sns.set(style="whitegrid")

# Class distribution bar plot
plt.figure(figsize=(6,4))
sns.countplot(x="multiclass_label", data=df, palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Class (-1 = Neither, 0 = MH only, 1 = Pain only, 2 = Both)")
plt.ylabel("Count")
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df["approx_age"], bins=40, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Length of stay
plt.figure(figsize=(6,4))
sns.histplot(df["length_of_stay"], bins=40, kde=False)
plt.title("Length of Stay (days)")
plt.xlabel("Days")
plt.ylabel("Count")
plt.xlim(0, 60)  # zoom in for readability
plt.show()

# Insurance group
plt.figure(figsize=(6,4))
sns.countplot(y="insurance_group", data=df, order=df["insurance_group"].value_counts().index)
plt.title("Insurance Groups")
plt.xlabel("Count")
plt.ylabel("Insurance Type")
plt.show()

# Polypharmacy flag
plt.figure(figsize=(6,4))
sns.countplot(x="polypharmacy_flag", data=df)
plt.title("Polypharmacy (5+ meds)")
plt.xlabel("Flag")
plt.ylabel("Count")
plt.show()

# ICU exposure
plt.figure(figsize=(6,4))
sns.countplot(x="was_in_icu", data=df)
plt.title("ICU Exposure")
plt.xlabel("In ICU")
plt.ylabel("Count")
plt.show()

# ---------------------------------------------------------------------
# 4. Cross-feature quick check
# ---------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x="multiclass_label", y="approx_age", data=df)
plt.title("Age by Class")
plt.xlabel("Class")
plt.ylabel("Age")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="multiclass_label", y="length_of_stay", data=df)
plt.title("Length of Stay by Class")
plt.xlabel("Class")
plt.ylabel("Days")
plt.show()

print("\n✅ EDA complete. Review the plots for anomalies.")
