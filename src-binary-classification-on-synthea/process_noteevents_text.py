import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. Load Medications Data
# ---------------------------------------------------------------------
medications_path = os.getenv("synthea_medications_path", "./synthea_data/medications.csv")
medications_df = pd.read_csv(medications_path, low_memory=False)
medications_df.columns = medications_df.columns.str.lower().str.strip()

# Aggregate descriptions per patient
medications_text = medications_df.groupby("patient")["description"].apply(lambda x: " ".join(x)).reset_index()

# Rename column for clarity
agg_notes = medications_text.rename(columns={"description": "TEXT"})

#----------------------------------------------------------------------
# 2. Create note sequences
# ---------------------------------------------------------------------
note_sequences = {}
for patient, group in medications_text.groupby('patient'):
    clean_notes = group["description"].tolist()
    note_sequences.setdefault(patient, []).append(clean_notes)

# ---------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------
out_path = "./note_sequences_per_patient.npy"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.save(out_path, note_sequences)

# ---------------------------------------------------------------------
# 4. Stats
# ---------------------------------------------------------------------
note_lengths = [len(adm) for subj in note_sequences.values() for adm in subj]
print(f"\nSaved cleaned note sequences → {out_path}")
print(f"Total patients:      {len(note_sequences):,}")
