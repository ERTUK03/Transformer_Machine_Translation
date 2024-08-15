import torch

class Embeddings(torch.nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__(vocab_size, d_model)
