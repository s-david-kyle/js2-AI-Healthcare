import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# 1. Database Connection
# ---------------------------------------------------------------------
HOST = os.getenv("MIMIC_HOST", "localhost")
DBNAME = os.getenv("MIMIC_DBNAME", "mimic")
USER = os.getenv("MIMIC_USER", "")
PASSWORD = os.getenv("MIMIC_PASSWORD", "") 
SCHEMA = os.getenv("MIMIC_SCHEMA", "")
PORT = int(os.getenv("MIMIC_PORT", 5432))

DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text(f"SET search_path TO {SCHEMA};"))

print("Connected to the MIMIC-III database.")

# ---------------------------------------------------------------------
# 2. Define ICD-9 Code Sets + helper
# ---------------------------------------------------------------------
MENTAL_HEALTH_CODES = {
    'depression': ['2962', '2963', '311'],
    'bipolar': ['2964', '2965', '2966', '2967'],
    'anxiety': ['3000', '3002', '3003'],
    'ptsd': ['30981'],
    'psychotic': ['295', '297', '298'],
    'personality': ['301'],
    'substance_use': ['303', '304', '305']
}

CHRONIC_PAIN_CODES = {
    'chronic_pain_general': ['3382', '3384'],
    'back_pain': ['724'],
    'arthritis': ['714', '715'],
    'fibromyalgia': ['7291'],
    'migraine': ['346'],
    'headache': ['7840'],
    'diabetic_neuropathy': ['2506', '3572']
}

MENTAL_HEALTH_SET = {c for codes in MENTAL_HEALTH_CODES.values() for c in codes}
CHRONIC_PAIN_SET = {c for codes in CHRONIC_PAIN_CODES.values() for c in codes}

def normalize_icd9(code: str) -> str:
    return code.replace('.', '').lstrip('0')  # remove dots, strip leading zeros

def matches_any(code, code_set):
    code = normalize_icd9(str(code))
    return any(code.startswith(prefix) for prefix in code_set)


# ---------------------------------------------------------------------
# 3. Load Tables
# ---------------------------------------------------------------------
print("Loading tables...")

# ---------------------------------------------------------------------
# 3. Load Tables (schema-qualified)
# ---------------------------------------------------------------------
print("Loading tables...")

diag_df = pd.read_sql(f"""
    SELECT subject_id, hadm_id, icd9_code
    FROM {SCHEMA}.diagnoses_icd
    WHERE icd9_code IS NOT NULL;
""", engine)

adm_df = pd.read_sql(f"""
    SELECT subject_id, hadm_id, admittime, dischtime, insurance, admission_type
    FROM {SCHEMA}.admissions;
""", engine)

pat_df = pd.read_sql(f"""
    SELECT subject_id, gender, dob, dod, expire_flag
    FROM {SCHEMA}.patients;
""", engine)

transfers_df = pd.read_sql(f"""
    SELECT subject_id, hadm_id, icustay_id, intime, outtime, curr_careunit
    FROM {SCHEMA}.transfers;
""", engine)

services_df = pd.read_sql(f"""
    SELECT subject_id, hadm_id, curr_service
    FROM {SCHEMA}.services;
""", engine)

prescriptions_df = pd.read_sql(f"""
    SELECT subject_id, hadm_id, drug
    FROM {SCHEMA}.prescriptions;
""", engine)

print("All tables loaded.")

print(f"📊 diagnoses_icd rows: {len(diag_df)}")
print(f"📊 admissions rows: {len(adm_df)}")
print(f"📊 patients rows: {len(pat_df)}")
print(f"📊 transfers rows: {len(transfers_df)}")
print(f"📊 services rows: {len(services_df)}")
print(f"📊 prescriptions rows: {len(prescriptions_df)}")
print("Checking depression example:")
print([c for c in diag_df['icd9_code'].unique() if str(c).startswith("2962")][:10])

# ---------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------
print("Processing features...")

# Diagnosis aggregation
diag_agg = diag_df.groupby(['subject_id', 'hadm_id'])['icd9_code'].apply(list).reset_index()
diag_agg['has_mental_health'] = diag_agg['icd9_code'].apply(lambda codes: any(matches_any(c, MENTAL_HEALTH_SET) for c in codes))
diag_agg['has_chronic_pain'] = diag_agg['icd9_code'].apply(lambda codes: any(matches_any(c, CHRONIC_PAIN_SET) for c in codes))
diag_agg['diagnosis_count'] = diag_agg['icd9_code'].apply(len)

# Multiclass label assignment
def assign_multiclass_label(row):
    if row['has_mental_health'] and row['has_chronic_pain']:
        return 2
    elif row['has_chronic_pain']:
        return 1
    elif row['has_mental_health']:
        return 0
    else:
        return -1  # Explicit class for neither condition

diag_agg['multiclass_label'] = diag_agg.apply(assign_multiclass_label, axis=1)

# Merge tables
merged_df = (
    diag_agg.merge(adm_df, on=['subject_id', 'hadm_id'], how='left')
            .merge(pat_df, on='subject_id', how='left')
            .merge(services_df, on=['subject_id', 'hadm_id'], how='left')
)

# Additional features
# Safe datetime conversion
admit_times = pd.to_datetime(merged_df['admittime'], errors='coerce')
dob_times   = pd.to_datetime(merged_df['dob'], errors='coerce')
disch_times = pd.to_datetime(merged_df['dischtime'], errors='coerce')
dod_times   = pd.to_datetime(merged_df['dod'], errors='coerce')

# ---- Safe functions ----
def safe_age(admit, dob):
    try:
        if pd.isna(admit) or pd.isna(dob):
            return np.nan
        dob_str = str(dob)
        if dob_str.startswith("2100"):   # de-identified DOB for >89 yrs
            return 90
        # Convert to Python datetime to avoid Pandas overflow
        admit_dt = admit.to_pydatetime() if hasattr(admit, "to_pydatetime") else admit
        dob_dt   = dob.to_pydatetime() if hasattr(dob, "to_pydatetime") else dob
        age = (admit_dt - dob_dt).days / 365.25
        if age < 0 or age > 120:
            return np.nan
        return age
    except Exception:
        return np.nan

def safe_los(admit, disch):
    try:
        if pd.isna(admit) or pd.isna(disch):
            return np.nan
        admit_dt = admit.to_pydatetime() if hasattr(admit, "to_pydatetime") else admit
        disch_dt = disch.to_pydatetime() if hasattr(disch, "to_pydatetime") else disch
        los = (disch_dt - admit_dt).days
        if los < 0 or los > 365:  # sanity cap 1 year
            return np.nan
        return los
    except Exception:
        return np.nan

def safe_in_hospital_mortality(admit, disch, dod):
    try:
        if pd.isna(admit) or pd.isna(disch) or pd.isna(dod):
            return False
        dod_str = str(dod)
        if dod_str.startswith("2100"):   # fake placeholder
            return False
        admit_dt = admit.to_pydatetime() if hasattr(admit, "to_pydatetime") else admit
        disch_dt = disch.to_pydatetime() if hasattr(disch, "to_pydatetime") else disch
        dod_dt   = dod.to_pydatetime() if hasattr(dod, "to_pydatetime") else dod
        return admit_dt <= dod_dt <= disch_dt
    except Exception:
        return False


# ---- Apply safely ----
merged_df['approx_age'] = [
    safe_age(a, d) for a, d in zip(admit_times, dob_times)
]
merged_df['length_of_stay'] = [
    safe_los(a, d) for a, d in zip(admit_times, disch_times)
]
merged_df['in_hospital_mortality'] = [
    safe_in_hospital_mortality(a, d, dod) for a, d, dod in zip(admit_times, disch_times, dod_times)
]


# Insurance simplification
def simplify_insurance(ins):
    ins = ins.upper()
    if 'MEDICARE' in ins:
        return 'Medicare'
    elif 'MEDICAID' in ins:
        return 'Medicaid'
    elif 'PRIVATE' in ins or 'COMMERCIAL' in ins or 'HMO' in ins:
        return 'Private'
    elif 'GOVERNMENT' in ins:
        return 'Government'
    elif 'SELF PAY' in ins:
        return 'Self Pay'
    else:
        return 'Other'

merged_df['insurance_group'] = merged_df['insurance'].fillna('UNKNOWN').apply(simplify_insurance)

# ---------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------
PSYCH_MEDICATIONS = [
    'prozac', 'sertraline', 'zoloft', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram', 
    'venlafaxine', 'duloxetine', 'bupropion', 'amitriptyline', 'nortriptyline', 'trazodone',
    'olanzapine', 'risperidone', 'quetiapine', 'aripiprazole', 'haloperidol', 'lithium',
    'diazepam', 'lorazepam', 'alprazolam', 'clonazepam', 'buspirone'
]

PAIN_MEDICATIONS = [
    'morphine', 'oxycodone', 'hydrocodone', 'fentanyl', 'codeine', 'tramadol', 
    'gabapentin', 'pregabalin', 'lidocaine', 'methadone', 'buprenorphine',
    'ibuprofen', 'naproxen', 'celecoxib', 'diclofenac'
]

MEDICATION_REGEX = r'\b(?:' + '|'.join([med.lower() for med in (PSYCH_MEDICATIONS + PAIN_MEDICATIONS)]) + r')\b'

filtered_rx = prescriptions_df[
    prescriptions_df['drug'].str.lower().str.contains(MEDICATION_REGEX, na=False, regex=True)
][['subject_id', 'hadm_id']].drop_duplicates()

filtered_rx['on_psych_or_pain_meds'] = True
merged_df = merged_df.merge(filtered_rx, on=['subject_id', 'hadm_id'], how='left')
merged_df['on_psych_or_pain_meds'] = merged_df['on_psych_or_pain_meds'].fillna(False)

# Medication counts
medication_count = prescriptions_df.groupby(['subject_id', 'hadm_id']).size().reset_index(name='medication_count')
psych_or_pain_rx_count = filtered_rx.groupby(['subject_id', 'hadm_id']).size().reset_index(name='psych_or_pain_rx_count')

merged_df = merged_df.merge(medication_count, on=['subject_id', 'hadm_id'], how='left')
merged_df = merged_df.merge(psych_or_pain_rx_count, on=['subject_id', 'hadm_id'], how='left')

merged_df['medication_count'] = merged_df['medication_count'].fillna(0)
merged_df['psych_or_pain_rx_count'] = merged_df['psych_or_pain_rx_count'].fillna(0)
merged_df['polypharmacy_flag'] = (merged_df['medication_count'] >= 5).astype(int)

# ---------------------------------------------------------------------
# Transfers / ICU features
# ---------------------------------------------------------------------
transfers_df['curr_careunit'] = transfers_df['curr_careunit'].str.upper()
transfers_df['was_in_icu'] = transfers_df['curr_careunit'].str.contains('ICU|CCU|MICU|SICU', na=False)

was_in_icu = transfers_df.groupby(['subject_id', 'hadm_id'])['was_in_icu'].max().reset_index()
transfer_count = transfers_df.groupby(['subject_id', 'hadm_id']).size().reset_index(name='transfer_count')

merged_df = merged_df.merge(was_in_icu, on=['subject_id', 'hadm_id'], how='left')
merged_df = merged_df.merge(transfer_count, on=['subject_id', 'hadm_id'], how='left')

merged_df['was_in_icu'] = merged_df['was_in_icu'].fillna(False)
merged_df['transfer_count'] = merged_df['transfer_count'].fillna(0)

# Event sequence as JSON
transfers_df['intime'] = pd.to_datetime(transfers_df['intime'])
event_sequence = (
    transfers_df.sort_values(['subject_id', 'hadm_id', 'intime'])
    .groupby(['subject_id', 'hadm_id'])[['curr_careunit', 'intime']]
    .apply(lambda df: list(zip(df['curr_careunit'], df['intime'].astype(str))))
    .reset_index()
    .rename(columns={0: 'curr_careunit_sequence'})
)

# Safe merge
merged_df = merged_df.merge(event_sequence, on=['subject_id', 'hadm_id'], how='left')

if 'curr_careunit_sequence' not in merged_df.columns:
    merged_df['curr_careunit_sequence'] = [[] for _ in range(len(merged_df))]

# Ensure NaNs are replaced with empty list
merged_df['curr_careunit_sequence'] = merged_df['curr_careunit_sequence'].apply(
    lambda x: x if isinstance(x, list) else []
)

# Dump to JSON for saving
merged_df['curr_careunit_sequence'] = merged_df['curr_careunit_sequence'].apply(json.dumps)

# Co-occurrence flag
merged_df['co_occurrence'] = (
    merged_df['has_mental_health'] & merged_df['has_chronic_pain']
)

# Seen by psych (service string contains "PSY")
if 'curr_service' not in merged_df.columns:
    merged_df['curr_service'] = ""   # ensure column exists
merged_df['seen_by_psych'] = merged_df['curr_service'].fillna('').str.contains("PSY", case=False)

# ---------------------------------------------------------------------
# 5. Save Output
# ---------------------------------------------------------------------
final_cols = [
    'subject_id', 'hadm_id', 'icd9_code', 'diagnosis_count', 'has_mental_health',
    'has_chronic_pain', 'co_occurrence', 'multiclass_label', 'insurance', 'admission_type', 'gender',
    'approx_age', 'admittime', 'dischtime', 'length_of_stay', 'curr_careunit_sequence',
    'curr_service', 'seen_by_psych', 'on_psych_or_pain_meds', 'medication_count',
    'was_in_icu', 'in_hospital_mortality', 'expire_flag', 'transfer_count', 'insurance_group',
    'polypharmacy_flag', 'psych_or_pain_rx_count',
]

output_csv = "mimic_enriched_features.csv"
merged_df[final_cols].to_csv(output_csv, index=False)
print(f"Saved enriched dataset to {output_csv}.")

engine.dispose()
print("Database connection closed.")

# Class counts
counts = merged_df["multiclass_label"].value_counts(dropna=False).sort_index()
print("\n📊 Class Distribution Before Saving:")
for label in [-1, 0, 1, 2]:
    print(f"  Class {label}: {counts.get(label, 0)}")

summary = {
    "n_rows": len(merged_df),
    "class_counts": counts.to_dict(),
}
with open("mimic_enriched_features_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
