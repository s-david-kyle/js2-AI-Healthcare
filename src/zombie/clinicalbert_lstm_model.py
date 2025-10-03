import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# ClinicalBERT + Bidirectional LSTM + Attention Model Definition
# ---------------------------------------------------------------------
class ClinicalBERT_LSTM(nn.Module):
    def __init__(self, bert_model, structured_input_dim, hidden_dim=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.bert = bert_model  # Preloaded ClinicalBERT
        self.bert_proj = nn.Linear(768, hidden_dim)  # Reduce BERT dim to match structured input

        self.lstm = nn.LSTM(input_size=hidden_dim + structured_input_dim,  # fused input
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, structured_seq):
        """
        input_ids:     (B, T, 512)
        attention_mask:(B, T, 512)
        structured_seq:(B, T, F)
        """
        B, T, L = input_ids.shape  # L should be 512
        input_ids = input_ids.reshape(B * T, L)
        attention_mask = attention_mask.reshape(B * T, L)

        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state[:, 0, :]  # CLS

        bert_emb = self.bert_proj(bert_out)            # (B*T, hidden_dim)
        bert_emb = bert_emb.reshape(B, T, -1)          # (B, T, hidden_dim)

        if bert_emb.size(1) != structured_seq.size(1):
            T_common = min(bert_emb.size(1), structured_seq.size(1))
            bert_emb = bert_emb[:, -T_common:, :]
            structured_seq = structured_seq[:, -T_common:, :]

        fused = torch.cat([bert_emb, structured_seq], dim=-1)  # (B, T, H+F)
        lstm_out, _ = self.lstm(fused)                         # (B, T, 2*H)

        attn_scores = self.attn(lstm_out)                      # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)      # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)   # (B, 2*H)

        logits = self.classifier(context)                     # (B, 1)
        return logits.view(-1)                                # (B,)
