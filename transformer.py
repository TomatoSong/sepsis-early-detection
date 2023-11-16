import math
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FeatureExpander(nn.Module):
    def __init__(self, input_features, output_features):
        super(FeatureExpander, self).__init__()
        self.expander = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x should be of shape (batch_size, seq_length, input_features)
        batch_size, seq_length, input_features = x.size()

        # Reshape x to (batch_size * seq_length, input_features)
        x = x.contiguous().view(-1, input_features)

        # Apply the MLP
        x = self.expander(x)

        # Reshape x back to (batch_size, seq_length, output_features)
        x = x.contiguous().view(batch_size, seq_length, -1)

        return x

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        k = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, nheads, nlayers, ff_dim, nclasses):
        super(TimeSeriesClassifier, self).__init__()
        self.feature_expander = FeatureExpander(
            input_features=input_dim,
            output_features=hidden_dim
        )
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim,
            seq_len=seq_len
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x, mask):
        ### x : batch_size, sequence_length, num_features
        x = self.feature_expander(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, x, src_key_padding_mask=mask)
        x, _ = torch.max(x, 1)
        x = self.classifier(x)
        return x
