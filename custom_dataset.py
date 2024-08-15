from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, data_pairs):
        self.src = data_pairs[0]
        self.tgt = data_pairs[1]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx]), torch.tensor(self.tgt[idx])
