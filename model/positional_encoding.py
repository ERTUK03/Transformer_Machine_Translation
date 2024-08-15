import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.positions = torch.zeros(max_len, d_model).to(device)
        self.positions.requires_grad = False
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.positions[:, 0::2] = torch.sin(positions * div_term)
        self.positions[:, 1::2] = torch.cos(positions * div_term)

    def forward(self, x):
        return self.positions[:x.size(-1), :]
