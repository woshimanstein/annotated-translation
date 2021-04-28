import json
import os
import torch
from train_tagger import NER_tagger

model = NER_tagger()
model.load_state_dict(torch.load(os.path.join('model_weights', 'tagger.pt')))
word_to_idx = json.load(open('word_to_idx'))
tag_to_idx = json.load(open('tag_to_idx'))

TXT = ["transformer", "is", "a", "great", "deep", "learning", "model", "and", "so", "is", "rnn", "."]
input_ids = []
for word in TXT:
    if word in word_to_idx:
        input_ids.append(word_to_idx[word])
    else:
        input_ids.append(word_to_idx['[UNK]'])
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
print(input_ids)

logits = model(input_ids)
print(torch.argmax(logits, dim=2))