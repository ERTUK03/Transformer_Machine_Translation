from model.multi_head_attention import MultiHeadAttention
from model.position_feed_forward import PositionwiseFeedForward
import torch

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        masked_attn_output = self.masked_multi_head_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + masked_attn_output)
        attn_output = self.multi_head_attention(encoder_output, encoder_output, x, src_mask)
        x = self.layer_norm2(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + ff_output)
        return x
