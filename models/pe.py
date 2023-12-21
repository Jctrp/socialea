import torch.nn as nn
import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        
        self.pe = torch.zeros(max_len, emb_size, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        return self.pe[:, :x.size(1), :]
