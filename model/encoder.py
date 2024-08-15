from model.encoder_layer import EncoderLayer
import torch

class Encoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
