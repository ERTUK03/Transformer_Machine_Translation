from model.decoder_layer import DecoderLayer
import torch

class Decoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
