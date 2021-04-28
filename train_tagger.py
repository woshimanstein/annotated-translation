import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from data import MaskedNERDataset, MyCollate

BATCH_SIZE = 32
EPOCH = 5
LR = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

word_to_idx = json.load(open('word_to_idx'))
tag_to_idx = json.load(open('tag_to_idx'))

class NER_tagger(nn.Module):
    def __init__(self, vocab_size=18991, embed_dim=256, hidden_dim=128, dropout=0.1, padding_idx=0, num_tags=3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)
        self.out = nn.Linear(hidden_dim, num_tags)

    def forward(self, input_seq):
        '''
        Parameters
        -----------
        input_seq: torch.Tensor (batch, seq_len)

        Returns
        -----------
        torch.Tensor (batch, seq_len, num_tags)
            The logits of tags for each word in the input sequence
        '''

        lengths = torch.sum(input_seq != 0, dim=1).to('cpu')
        input_embedding = self.dropout(self.embedding(input_seq))
        input_packed = pack_padded_sequence(input_embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.rnn(input_packed)
        unpacked_output, _ = pad_packed_sequence(output, batch_first=True)
        logits = self.out(unpacked_output)
        return logits

def compute_accuracy(model, data='valid'):
    dataset = MaskedNERDataset(word_to_idx, tag_to_idx, data=data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=MyCollate(word_to_idx, tag_to_idx), shuffle=True)

    model.eval()
    with torch.no_grad():
        correct = 0
        num_tags = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for batch in dataloader:
            inputs = batch['sentence'].to(device)
            labels = batch['tag'].to(device)

            logits = model(inputs)
            pred = torch.argmax(logits, dim=2)

            # for tag-level accuracy
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i][j] != tag_to_idx['[pad]']:
                        num_tags += 1
                        if labels[i][j] == pred[i][j]:
                            correct += 1

            # for entity level precision
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if pred[i][j] == tag_to_idx['B']:
                        if labels[i][j] == tag_to_idx['B']:
                            same = True
                            k = j + 1
                            while same and k < labels.shape[1] and labels[i][k] == tag_to_idx['I']:
                                same = labels[i][k] == pred[i][k]
                                k += 1
                            if same:
                                TP += 1
                            else:
                                FP += 1
                            
                            if k < labels.shape[1]:
                                j = k
                            else:
                                break

        print(f"Accuracy in {data} set: {correct / num_tags}")
        print(f"Precision in {data} set: {TP / (TP + FP)}")

    model.train()

if __name__ == '__main__':

    train_dataset = MaskedNERDataset(word_to_idx, tag_to_idx)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=MyCollate(word_to_idx, tag_to_idx), shuffle=True)

    model = NER_tagger().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    for epo in range(EPOCH):
        itr = 0
        total_loss = 0
        for batch in train_dataloader:
            # zero-out gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch['sentence'].to(device)
            labels = batch['tag'].to(device)
            logits = model(inputs)


            # compute loss and perfom a step
            logits = logits.reshape(-1, logits.shape[-1]).to(device)
            labels = labels.reshape(-1).to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
            itr += 1
            total_loss += loss.item()
        
        print(f"Average Loss in epoch {epo}: {total_loss / itr}")
        # compute_accuracy(model)


    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('model_weights', 'tagger.pt'))
        