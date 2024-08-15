from model.multi_head_attention import MultiHeadAttention
from model.position_feed_forward import PositionwiseFeedForward
import torch

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x
