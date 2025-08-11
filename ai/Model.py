import torch
import torch.nn as nn


class MaskAwareAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=2, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x, mask):
        x_proc = torch.nan_to_num(x, nan=0.0)
        lstm_in = torch.cat([x_proc, mask], dim=2)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        output = self.head(attn_out[:, -1])
        return output

