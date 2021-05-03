import os
import torch
from torch.utils.data import Dataset, DataLoader

class TranslationData(Dataset):
    def __init__(self, data='train'):
        super().__init__()
        with open(os.path.join('translation_data', f'annotated_de-en_{data}.tsv'), 'r') as data_file:
            lines = data_file.readlines()
            self.lines_without_empty = []
            for line in lines:
                if line.strip() != '' and len(line.strip().split('\t')) == 2:
                    self.lines_without_empty.append(line.strip())
        
    def __len__(self):
        return len(self.lines_without_empty)

    def __getitem__(self, index):
        pair = self.lines_without_empty[index].split('\t')
        return {'de': pair[0], 'en': pair[1]}

if __name__ == '__main__':
    dataloader = DataLoader(TranslationData(), batch_size=2)
    print(len(dataloader))
    batch = next(iter(dataloader))
    print(batch)
