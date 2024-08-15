import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        k_t = k.transpose(2, 3)
        attn_scores = torch.matmul(q, k_t) / math.sqrt(k.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.nn.functional.softmax(attn_scores,dim=-1)
        output = torch.matmul(attn_probs, v)
        return output
