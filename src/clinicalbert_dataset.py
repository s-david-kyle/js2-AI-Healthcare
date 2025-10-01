import torch
from torch.utils.data import Dataset

class ClinicalBERTDataset(Dataset):
    """
    Dataset for ClinicalBERT + Transformer model.

    Accepts tokenized notes per patient as list/dict/tuple.
    Structured data is a sequence aligned with notes.
    """

    def __init__(self, X_structured, y, tokenized_notes):
        # Normalize tokenized_notes to list if dict
        if isinstance(tokenized_notes, dict):
            tokenized_notes = list(tokenized_notes.values())
        tokenized_notes = list(tokenized_notes)

        # Align lengths
        N = min(len(X_structured), len(y), len(tokenized_notes))
        self.X_structured = X_structured[:N]
        self.y = y[:N]
        self.notes = tokenized_notes[:N]

    def __len__(self):
        return len(self.y)

    def _unpack_note(self, item):
        # Case A: tuple/list of tensors
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], torch.Tensor):
            return item[0], item[1]

        # Case B: dict with tensors
        if isinstance(item, dict) and "input_ids" in item:
            return item["input_ids"], item["attention_mask"]

        # Case C: list of sub-items (multiple notes)
        if isinstance(item, list):
            ids_list, mask_list = [], []
            for sub in item:
                ids, mask = self._unpack_note(sub)
                ids_list.append(ids)
                mask_list.append(mask)
            return torch.stack(ids_list), torch.stack(mask_list)

        raise TypeError("Unsupported tokenized note format")

    def __getitem__(self, idx):
        struct_seq = torch.tensor(self.X_structured[idx], dtype=torch.float32)  # shape (T, F)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        input_ids, attention_mask = self._unpack_note(self.notes[idx])  # (T, L), (T, L)
        return input_ids, attention_mask, struct_seq, label
