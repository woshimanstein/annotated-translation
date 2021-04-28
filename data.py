import os
import json
import random
import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def load_tag_data(tag_file):
    all_sentences = []
    all_tags = []
    sent = []
    tags = []
    with open(tag_file, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            if line.strip() == "":
                all_sentences.append(sent)
                all_tags.append(tags)
                sent = []
                tags = []
            else:
                word = line.strip().split()[0]
                tag = line.strip().split()[-1]
                sent.append(word)
                tags.append(tag)
    for i in range(len(all_tags)):
        for j in range(len(all_tags[i])):
            if all_tags[i][j].startswith('B'):
                all_tags[i][j] = 'B'
            elif all_tags[i][j].startswith('I'):
                all_tags[i][j] = 'I'

    return all_sentences, all_tags

def build_masked_data(ner_sentences, ner_tags):
    all_sentences = []
    all_tags = []
    for i in range(len(ner_sentences)):
        num_masked = 0
        non_entities = set()
        for j in range(len(ner_tags[i])):
            if ner_tags[i][j] == 'O':
                non_entities.add(j)
            else:
                num_masked += 1
                masked_sentence = copy.deepcopy(ner_sentences[i])
                masked_sentence[j] = '[MASK]'
                k = j + 1
                while k < len(ner_tags[i]) and ner_tags[i][k].startswith('I'):
                    masked_sentence[k] = '[MASK]'
                    k += 1
                all_sentences.append(masked_sentence)
                all_tags.append(copy.deepcopy(ner_tags[i]))

        sample = random.sample(non_entities, min(num_masked, len(non_entities)))
        for index in sample:
            masked_sentence = copy.deepcopy(ner_sentences[i])
            masked_sentence[index] = '[MASK]'
            all_sentences.append(masked_sentence)
            all_tags.append(copy.deepcopy(ner_tags[i]))
    return all_sentences, all_tags

class MaskedNERDataset(Dataset):
    def __init__(self, word_to_idx, tag_to_idx, data='train'):
        super().__init__()
        original_sentences, original_tags = load_tag_data(os.path.join('NER_data', f'{data}.txt'))
        self.sentences, self.tags = build_masked_data(original_sentences, original_tags)
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                lower_case = self.sentences[i][j].lower()
                if lower_case not in word_to_idx:
                    self.sentences[i][j] = word_to_idx['[UNK]']
                else:
                    self.sentences[i][j] = word_to_idx[lower_case]
        for i in range(len(self.tags)):
            for j in range(len(self.tags[i])):
                self.tags[i][j] = tag_to_idx[self.tags[i][j]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return {'sentence': self.sentences[index], 'tag': self.tags[index]}

class MyCollate:
    def __init__(self, word_to_idx, tag_to_idx):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

    def __call__(self, batch):
        sentences = [torch.tensor(data['sentence'], dtype=torch.long) for data in batch]
        tags = [torch.tensor(data['tag'], dtype=torch.long) for data in batch]
        sentences = pad_sequence(sentences, batch_first=True, padding_value=self.word_to_idx['[pad]'])
        tags = pad_sequence(tags, batch_first=True, padding_value=self.tag_to_idx['[pad]'])
        return {'sentence': sentences, 'tag': tags}


if __name__ == '__main__':
    original_sentences, original_tags = load_tag_data(os.path.join('NER_data', 'train.txt'))
    train_sentences, train_tags = build_masked_data(original_sentences, original_tags)

    unique_tags = set([tag for tag_seq in train_tags for tag in tag_seq])

    word_to_idx = {'[pad]': 0, '[MASK]': 1, '[UNK]': 2}
    for sent in train_sentences:
        for word in sent:
            if word.lower() not in word_to_idx:
                word_to_idx[word.lower()] = len(word_to_idx)

    idx_to_word = {}
    for word in word_to_idx:
        idx_to_word[word_to_idx[word]] = word
                
    tag_to_idx = {}
    for tag in unique_tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)
    tag_to_idx['[pad]'] = len(tag_to_idx)

    idx_to_tag = {}
    for tag in tag_to_idx:
        idx_to_tag[tag_to_idx[tag]] = tag

    json.dump(word_to_idx, open('word_to_idx', 'w'))
    json.dump(idx_to_word, open('idx_to_word', 'w'))
    json.dump(tag_to_idx, open('tag_to_idx', 'w'))
    json.dump(idx_to_tag, open('idx_to_tag', 'w'))
