from model.scaled_dot_product_attention import ScaledDotProductAttention
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)

        self.linear_out = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, q.size(-1) // self.num_heads)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, k.size(-1) // self.num_heads)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, v.size(-1) // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = self.scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, out.size(2), out.size(1)*out.size(3))

        output = self.linear_out(out)
        output = self.dropout(output)

        return output
