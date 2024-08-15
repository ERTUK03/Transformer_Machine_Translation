from model.embedding import Embeddings
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder
import torch

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, max_len, device, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.encoder = Encoder(d_model, d_ff, num_heads, num_layers, dropout)
        self.decoder = Decoder(d_model, d_ff, num_heads, num_layers, dropout)
        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, src, tgt):
        src_mask = self.generate_mask(src)
        tgt_mask = self.generate_mask(tgt)

        src = src.long()
        tgt = tgt.long()

        src_embeddings = self.embedding(src)
        tgt_embeddings = self.embedding(tgt)
        src_embeddings = self.dropout_1(src_embeddings + self.positional_encoding(src))
        tgt_embeddings = self.dropout_2(tgt_embeddings + self.positional_encoding(tgt))

        encoder_output = self.encoder(src_embeddings, src_mask)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        output = self.softmax(output)
        return output

    def generate_mask(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        return mask
