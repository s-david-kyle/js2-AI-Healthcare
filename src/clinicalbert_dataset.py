# clinicalbert_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_VISITS = 10  # keep consistent everywhere

# -------------------------------
# Raw tokenized notes dataset
# -------------------------------
class ClinicalBERTFastDatasetWithIDs(Dataset):
    """
    Dataset for ClinicalBERT with raw tokenized notes.
    Expects input_ids + attention_masks per visit.
    """
    def __init__(self, X_structured, y, input_ids, attn_masks, masks, subject_ids, max_visits=MAX_VISITS):
        N = min(len(X_structured), len(y), len(input_ids), len(attn_masks), len(masks), len(subject_ids))
        self.max_visits = max_visits or X_structured.shape[1]

        self.X_structured = torch.tensor(X_structured[:N], dtype=torch.float32)
        self.y = torch.tensor(y[:N], dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids[:N], dtype=torch.long)

        # Pre-pad visit masks
        padded_masks = np.zeros((N, self.max_visits), dtype=bool)
        for i, m in enumerate(masks[:N]):
            L = min(len(m), self.max_visits)
            padded_masks[i, :L] = m[:L].astype(bool)
        self.masks = torch.tensor(padded_masks, dtype=torch.bool)

        # Notes
        self.input_ids = torch.tensor(input_ids[:N], dtype=torch.long)
        self.attn_masks = torch.tensor(attn_masks[:N] > 0, dtype=torch.bool)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],      
            self.attn_masks[idx],     
            self.X_structured[idx],   
            self.masks[idx],          
            self.y[idx],              
            self.subject_ids[idx],    
        )

def collate_fn(batch):
    input_ids, attention_masks, structured, visit_masks, labels, subj_ids = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(structured),
        torch.stack(visit_masks),
        torch.stack(labels),
        torch.stack(subj_ids),
    )

# -------------------------------
# Precomputed embeddings dataset
# -------------------------------
class ClinicalBERTPrecomputedDataset(Dataset):
    """
    Dataset when using precomputed ClinicalBERT CLS embeddings.
    Expects bert_embs of shape (N, T, 768).
    """
    def __init__(self, X_structured, y, bert_embs, masks, subject_ids, max_visits=MAX_VISITS):
        N = min(len(X_structured), len(y), len(bert_embs), len(masks), len(subject_ids))
        self.max_visits = max_visits or X_structured.shape[1]

        self.X_structured = torch.tensor(X_structured[:N], dtype=torch.float32)
        self.y = torch.tensor(y[:N], dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids[:N], dtype=torch.long)

        padded_masks = np.zeros((N, self.max_visits), dtype=bool)
        for i, m in enumerate(masks[:N]):
            L = min(len(m), self.max_visits)
            padded_masks[i, :L] = m[:L].astype(bool)
        self.masks = torch.tensor(padded_masks, dtype=torch.bool)

        self.bert_embs = torch.tensor(bert_embs[:N], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.bert_embs[idx],      
            self.X_structured[idx],   
            self.masks[idx],          
            self.y[idx],              
            self.subject_ids[idx],    
        )

def collate_precomputed(batch):
    bert_embs, structured, visit_masks, labels, subj_ids = zip(*batch)
    return (
        torch.stack(bert_embs),
        torch.stack(structured),
        torch.stack(visit_masks),
        torch.stack(labels),
        torch.stack(subj_ids),
    )
