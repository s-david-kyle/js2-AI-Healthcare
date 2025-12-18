# boosted_mimiciii_text_features.py
# ---------------------------------------------------------------------
# Feature extraction script for MIMIC-III note-derived features
# Adds TF-IDF, sentiment, topic modeling, PCA/UMAP, clustering
# ---------------------------------------------------------------------

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
from textblob import TextBlob
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

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

# ---------------------------------------------------------------------
# 2. TF-IDF for Target Terms
# ---------------------------------------------------------------------
target_terms = [
    "opioid", "acetaminophen", "ibuprofen", "gabapentin", "morphine",
    "tramadol", "oxycodone", "hydrocodone", "meperidine", "fentanyl",
    "pregabalin", "naproxen"
]

vectorizer = TfidfVectorizer(vocabulary=target_terms)
tfidf_matrix = vectorizer.fit_transform(agg_notes['TEXT'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{term}" for term in target_terms])

# ---------------------------------------------------------------------
# 3. Sentiment Analysis
# ---------------------------------------------------------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
agg_notes['sentiment'] = agg_notes['TEXT'].apply(get_sentiment)

# ---------------------------------------------------------------------
# 4. Topic Modeling (LDA)
# ---------------------------------------------------------------------
count_vectorizer = CountVectorizer(max_df=0.95, min_df=10, stop_words='english')
counts = count_vectorizer.fit_transform(agg_notes['TEXT'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(counts)
topics_df = pd.DataFrame(topics, columns=[f"topic_{i+1}" for i in range(topics.shape[1])])

# ---------------------------------------------------------------------
# 5. Dimensionality Reduction
# ---------------------------------------------------------------------
text_feature_matrix = pd.concat([tfidf_df, topics_df], axis=1)
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(text_feature_matrix)

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_matrix)
pca_df = pd.DataFrame(pca_components, columns=["pca_1", "pca_2"])

umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3, metric='cosine')
umap_components = umap_model.fit_transform(scaled_matrix)
umap_df = pd.DataFrame(umap_components, columns=["umap_1", "umap_2"])

# ---------------------------------------------------------------------
# 6. Clustering
# ---------------------------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_matrix)
cluster_df = pd.DataFrame({'note_cluster': kmeans_labels})

# ---------------------------------------------------------------------
# 7. Combine All Note Features
# ---------------------------------------------------------------------
medications_description_features = pd.concat([
    agg_notes[['patient', 'sentiment']],
    tfidf_df,
    topics_df,
    pca_df,
    umap_df,
    cluster_df
], axis=1)

# ---------------------------------------------------------------------
# 8. Merge with Boosted Structured Features
# ---------------------------------------------------------------------
features_path = "synthea_enriched_features.csv"
features_df = pd.read_csv(features_path)

final_df = features_df.merge(medications_description_features, left_on='id', right_on='patient', how='left')
final_df.drop(columns='patient', inplace=True)

# ---------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------
print("Saving final dataset...")
final_df.to_csv("synthea_enriched_features_w_notes.csv", index=False)
print("✅ Saved final dataset with structured + note features → synthea_enriched_features_w_notes.csv")

# ---------------------------------------------------------------------
# 10. Save tokenization sequences for ClinicalBERT
# ---------------------------------------------------------------------
medications_sequences = {}
for patient, group in medications_text.groupby('patient'):
    clean_notes = group["description"].tolist()
    medications_sequences.setdefault(patient, []).append(clean_notes)

np.save("note_sequences_per_patient.npy", medications_sequences)
print("✅ Saved note_sequences_per_patient.npy for ClinicalBERT.")

# ---------------------------------------------------------------------
# 11. Save completion marker
# ---------------------------------------------------------------------
with open("./boosted_features_complete.txt", "w") as f:
    f.write("Boosted structured + medications features completed successfully.\n")
