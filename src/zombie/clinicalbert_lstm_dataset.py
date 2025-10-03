import torch
from torch.utils.data import Dataset

class ClinicalBERTLSTMDataset(Dataset):
    """Robust dataset for ClinicalBERT‑LSTM fusion.

    Accepts many tokenization storage formats:
      • tuple  -> (input_ids, attention_mask)
      • dict   -> {"input_ids": tensor, "attention_mask": tensor}
      • list   -> list of the above (one per note); we stack along time axis
    """

    def __init__(self, X_structured, y, tokenized_notes, masks):
            if isinstance(tokenized_notes, dict):
                tokenized_notes = list(tokenized_notes.values())
            tokenized_notes = list(tokenized_notes)

            N = min(len(X_structured), len(y), len(tokenized_notes), len(masks))
            self.X_structured = X_structured[:N]
            self.y = y[:N]
            self.notes = tokenized_notes[:N]
            self.masks = masks[:N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        struct_seq = torch.tensor(self.X_structured[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        visit_mask = torch.tensor(self.masks[idx], dtype=torch.float32)  # (T,)

        input_ids, attention_mask = self._unpack_note(self.notes[idx])
        return input_ids, attention_mask, struct_seq, visit_mask, label

    # ------------------------------------------------------------
    # helper to unpack a single note item to (ids, mask) tensors
    # ------------------------------------------------------------
    def _unpack_note(self, item):
        # Case A: already a tuple/list of two tensors
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], torch.Tensor):
            return item[0], item[1]

        # Case B: single dict with tensors
        if isinstance(item, dict) and "input_ids" in item:
            return item["input_ids"], item["attention_mask"]

        # Case C: list of sub‑items (multiple notes)
        if isinstance(item, list):
            ids_list, mask_list = [], []
            for sub in item:
                ids, mask = self._unpack_note(sub)  # recurse
                ids_list.append(ids)
                mask_list.append(mask)
            return torch.stack(ids_list), torch.stack(mask_list)

        raise TypeError("Unsupported tokenized note format")

    def __getitem__(self, idx):
        struct_seq = torch.tensor(self.X_structured[idx], dtype=torch.float32)
        label      = torch.tensor(self.y[idx], dtype=torch.float32)
        input_ids, attention_mask = self._unpack_note(self.notes[idx])
        return input_ids, attention_mask, struct_seq, label