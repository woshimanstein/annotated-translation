import os
import sys
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer, get_linear_schedule_with_warmup
import datasets
sys.path.insert(0, os.path.abspath('..'))
from model.PointerGenerator import *
from preprocessing.translation_data import *

os.chdir('../')

EVAL_BATCH_SIZE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = PointerGenerator(embed_size=128,
                        hidden_size=128,
                        num_layers=2).to(device)
MODEL_NAME = 'pointer_generator_4_epoch_5'
model.load_state_dict(torch.load(os.path.join('model_weights', f'{MODEL_NAME}.pt'), map_location=device))

dataset = TranslationData(data='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)

batch_num = 0
total_loss = 0
# metric_bleu = datasets.load_metric('sacrebleu')

# model.eval()
# with torch.no_grad():
#     for batch in tqdm(dataloader):
#         # input encoding
#         input_encoding = tokenizer(batch['en'], return_tensors='pt', padding=True, truncation=True)
#         input_ids = input_encoding['input_ids']
#         input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

#         # target encoding
#         target_encoding = tokenizer(batch['de'], return_tensors='pt', padding=True, truncation=True)
#         target_ids = target_encoding['input_ids']
#         target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

#         # forward pass
#         outputs = model(x=input_ids, y=target_ids)  # outputs.shape: (target_len, batch_size, vocab_size)

#         # prepare labels for cross entropy by removing the first time stamp (<s>)
#         labels = target_ids[1:, :]  # shape: (target_len - 1, batch_size)
#         labels = labels.reshape(-1).to(device)  # shape: ((target_len - 1) * batch_size)

#         # prepare model predicts for cross entropy by removing the last timestamp and merge first two axes
#         outputs = outputs[:-1, ...]  # shape: (target_len - 1, batch_size, vocab_size)
#         outputs = outputs.reshape(-1, outputs.shape[-1]).to(device)
#         # shape: ((target_len - 1) * batch_size, vocab_size)

#         # compute loss and perform a step
#         criterion = nn.CrossEntropyLoss(ignore_index=1)  # ignore padding index
#         loss = criterion(outputs, labels)

#         total_loss += float(loss)
#         batch_num += 1

#         input_ids = torch.transpose(input_ids, 0, 1).to(device)
#         model_res_ids = []
#         for source in input_ids:
#             length = torch.sum(source != 1)
#             model_res_ids.append(model.generate(source.reshape(-1, 1)[:length]))
#         predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in model_res_ids]

#         tmp_predictions, tmp_targets = [], []
#         for prediction, target in zip(predictions, batch['de']):
#             if len(target) > 0:
#                 tmp_predictions.append(prediction)
#                 tmp_targets.append(target)
#         predictions, targets = tmp_predictions, tmp_targets
#         references = [[r] for r in targets]
#         metric_bleu.add_batch(predictions=predictions, references=references)

# score_bleu = metric_bleu.compute()
# perplexity = np.exp(total_loss / batch_num)
        
# print(f'BLEU: {round(score_bleu["score"], 1)} out of {round(100., 1)}')
# print(f'Perplexity: {perplexity}')

model.eval()
with torch.no_grad():
    '''
    DataLoader
    '''
    valid_dataset = TranslationData(data='dev')
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=True
    )

    batch_num = 0
    total_loss = 0
    for batch in tqdm(valid_data_loader):
        # input encoding
        input_encoding = tokenizer(batch['en'], return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids']
        input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

        # target encoding
        target_encoding = tokenizer(batch['de'], return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids']
        target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

        # forward pass
        outputs = model(x=input_ids, y=target_ids)  # outputs.shape: (target_len, batch_size, vocab_size)

        # prepare labels for cross entropy by removing the first time stamp (<s>)
        labels = target_ids[1:, :]  # shape: (target_len - 1, batch_size)
        labels = labels.reshape(-1).to(device)  # shape: ((target_len - 1) * batch_size)

        # prepare model predicts for cross entropy by removing the last timestamp and merge first two axes
        outputs = outputs[:-1, ...]  # shape: (target_len - 1, batch_size, vocab_size)
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(device)
        # shape: ((target_len - 1) * batch_size, vocab_size)

        # compute loss and perform a step
        criterion = nn.CrossEntropyLoss(ignore_index=1)  # ignore padding index
        loss = criterion(outputs, labels)

        total_loss += float(loss)
        batch_num += 1

    perplexity = np.exp(total_loss / batch_num)
    print(perplexity)