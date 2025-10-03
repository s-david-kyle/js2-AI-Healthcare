import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ClinicalBERT_Transformer(nn.Module):
    def __init__(self, structured_input_dim=None,
                 hidden_dim=128, nhead=8, num_layers=2,
                 dropout=0.3):
        super().__init__()

        # Project CLS embeddings
        self.bert_proj = nn.Linear(768, hidden_dim)

        # Structured projection
        if structured_input_dim is not None:
            self.struct_proj = nn.Linear(structured_input_dim, hidden_dim)
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.struct_proj = None
            self.fusion_proj = None

        # Transformer across visits
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, bert_embs, structured_seq=None, visit_mask=None):
        """
        bert_embs: (B, T, 768) CLS embeddings
        structured_seq: (B, T, F)
        visit_mask: (B, T)
        """
        bert_emb = self.bert_proj(bert_embs)  # (B, T, H)

        if self.struct_proj is not None and structured_seq is not None:
            struct_emb = self.struct_proj(structured_seq)  # (B, T, H)
            fused = torch.cat([bert_emb, struct_emb], dim=-1)
            fused = self.fusion_proj(fused)
        else:
            fused = bert_emb

        tr_out = self.transformer_encoder(fused)

        if visit_mask is not None:
            mask = visit_mask.unsqueeze(-1).to(tr_out.device)
            pooled = (tr_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = tr_out.mean(dim=1)

        return self.classifier(pooled)
