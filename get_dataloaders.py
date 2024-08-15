from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def get_dataloaders(tokenized_data_pairs, batch_size, train_size=1):
    dataset = CustomDataset(tokenized_data_pairs)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    return train_dataloader, test_dataloader
