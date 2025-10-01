import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ClinicalBERT_Transformer(nn.Module):
    def __init__(self, bert_model, structured_input_dim=None, hidden_dim=128, nhead=8, num_layers=2, dropout=0.3):
        """
        bert_model: pretrained ClinicalBERT model from transformers
        structured_input_dim: optionally concatenate structured data (if None, ignore)
        hidden_dim: embedding size for Transformer encoder
        nhead: number of attention heads in Transformer
        num_layers: number of Transformer encoder layers
        dropout: dropout rate
        """
        super().__init__()
        self.bert = bert_model  # pretrained ClinicalBERT
        self.dropout = nn.Dropout(dropout)

        # Project ClinicalBERT's CLS embedding from 768 to hidden_dim
        self.bert_proj = nn.Linear(768, hidden_dim)

        # Optional structured input processing
        if structured_input_dim is not None:
            self.struct_proj = nn.Linear(structured_input_dim, hidden_dim)
        else:
            self.struct_proj = None

        # Transformer Encoder Layers
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier: input is hidden_dim after Transformer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, structured_seq=None):
        """
        input_ids: (B, T, L) token ids (T=number of notes/time steps, L=seq length e.g., 512)
        attention_mask: (B, T, L)
        structured_seq: (B, T, F) optional structured features per note/time step

        Output:
          logits (B,)
        """

        B, T, L = input_ids.shape

        # Flatten batch and time dims to process each note with ClinicalBERT
        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L)

        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # CLS token

        bert_emb = self.bert_proj(bert_out)  # (B*T, hidden_dim)
        bert_emb = bert_emb.view(B, T, -1)  # (B, T, hidden_dim)

        # If structured features are provided, project and fuse them
        if self.struct_proj is not None and structured_seq is not None:
            # Align sequence lengths if needed
            if bert_emb.size(1) != structured_seq.size(1):
                T_common = min(bert_emb.size(1), structured_seq.size(1))
                bert_emb = bert_emb[:, :T_common, :]
                structured_seq = structured_seq[:, :T_common, :]

            struct_emb = self.struct_proj(structured_seq)  # (B, T, hidden_dim)
            combined = bert_emb + struct_emb               # simple fusion
        else:
            combined = bert_emb
          
        combined = self.dropout(combined)

        # Transformer expects (S, B, E) format: swap batch and sequence dims
        transformer_in = combined.transpose(0, 1)  # (T, B, hidden_dim)

        transformer_out = self.transformer_encoder(transformer_in)  # (T, B, hidden_dim)

        # Pool over sequence dim (T) - e.g. mean pooling
        pooled = transformer_out.mean(dim=0)  # (B, hidden_dim)

        logits = self.classifier(pooled)  # (B, 1)

        return logits #.view(-1)  # (B,)
