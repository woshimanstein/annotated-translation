import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as opt
from tqdm import tqdm
from transformers import BartTokenizer, get_linear_schedule_with_warmup
from Seq2Seq import *
from translation_data import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=10,
    help=f'Specify number of training epochs'
)
arg_parser.add_argument(
    '-b', '--batch',
    type=int,
    default=6,
    help=f'Specify batch size'
)
args = arg_parser.parse_args()

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(args.gpu)  # use an unoccupied GPU

# hyperparameter
NUM_EPOCH = args.epoch
BATCH_SIZE = args.batch
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

# model saving
os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
MODEL_NAME = f'seq2seq_{BATCH_SIZE}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')
print(f"training seq2seq with batch size {BATCH_SIZE} for {NUM_EPOCH} epochs")

# model setup
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = Seq2Seq(embed_size=EMBEDDING_DIM,
                hidden_size=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT).to(device)
optimizer = opt.Adam(model.parameters())

# record these for each epoch
loss_record = []
ppl_record = []
# training loop
for epo in range(2, NUM_EPOCH):
    model.train()
    total_loss = 0

    '''
    DataLoader
    '''
    dataset = TranslationData(data='dev')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # training
    train_iterator_with_progress = tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        # input encoding
        input_encoding = tokenizer(batch['en'], return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids']
        input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

        # target encoding
        target_encoding = tokenizer(batch['de'], return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids']
        target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

        # zero-out gradient
        optimizer.zero_grad()

        # forward pass
        outputs, _ = model(x=input_ids, y=target_ids)  # outputs.shape: (target_len, batch_size, vocab_size)

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

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # gradient clipping
        optimizer.step()
        # scheduler.step()

        # if idx % 1000 == 0:
        #     print(f'epoch: {epo}, batch: {idx}, memory reserved {torch.cuda.memory_reserved(DEVICE_ID) / 1e9} GB')
        #     print(f'epoch: {epo}, batch: {idx}, memory allocated {torch.cuda.memory_allocated(DEVICE_ID) / 1e9} GB')
        idx += 1

        total_loss += float(loss)
        train_iterator_with_progress.set_description(f'Epoch {epo}')
        train_iterator_with_progress.set_postfix({'Loss': loss.item()})

    loss_record.append(total_loss)
    print(f'Loss in epoch {epo}: {total_loss}')
    log_file.write(f'Epoch:{epo} ')
    log_file.write(f'Loss:{total_loss} ')

    # evaluation
    model.eval()
    with torch.no_grad():
        '''
        DataLoader
        '''
        valid_dataset = TranslationData(data='dev')
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        batch_num = 0
        total_loss = 0
        for batch in valid_data_loader:
            # input encoding
            input_encoding = tokenizer(batch['en'], return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids']
            input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

            # target encoding
            target_encoding = tokenizer(batch['de'], return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids']
            target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

            # forward pass
            outputs, _ = model(x=input_ids, y=target_ids)  # outputs.shape: (target_len, batch_size, vocab_size)

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
        ppl_record.append(perplexity)
        print(f'Perplexity: {perplexity}')
        log_file.write(f'Perplexity:{perplexity}\n')

    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)

# close log file
log_file.close()

# plot loss and ppl
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

epochs = list(range(NUM_EPOCH))
ax[0].plot(epochs, loss_record)
ax[0].set_title('Loss', fontsize=20)
ax[0].set_xlabel('Epoch', fontsize=15)
ax[0].set_ylabel('Loss', fontsize=15)

ax[1].plot(epochs, ppl_record)
ax[1].set_title('Perplexity', fontsize=20)
ax[1].set_xlabel('Epoch', fontsize=15)
ax[1].set_ylabel('Perplexity', fontsize=15)

os.makedirs(os.path.dirname('figures' + '/'), exist_ok=True)
fig.savefig(os.path.join('figures', f'{MODEL_NAME}'))


