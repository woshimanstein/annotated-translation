import torch
from transformers import BartTokenizer
from Seq2Seq import *

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
print(tokenizer.tokenize('Providing a price list helps elevate the conversation about priorities.'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# input encoding
input_encoding = tokenizer(['Wenn man eine Preisliste an der Hand hat, kann man die richtigen Prioritäten finden.'], return_tensors='pt', padding=True, truncation=True)
input_ids = input_encoding['input_ids']
input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

# target encoding
target_encoding = tokenizer(['Providing a price list helps elevate the conversation about priorities.'], return_tensors='pt', padding=True, truncation=True)
target_ids = target_encoding['input_ids']
target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

model = Seq2Seq(embed_size=128,
                hidden_size=128,
                num_layers=2).to(device)
model.load_state_dict(torch.load('model_weights/seq2seq_10_epoch_9.pt', map_location=device))
model.eval()
with torch.no_grad():
    outputs, attention_score = model(x=input_ids, y=target_ids)
    print(torch.argmax(attention_score, dim=1))
