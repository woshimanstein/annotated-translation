import os
import json
import nltk
import random
from tqdm import tqdm
import torch
import spacy
from transformers import BartTokenizer
import datasets
from Seq2Seq import *

spacy.prefer_gpu()
nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')

original_file = open(os.path.join('translation_data', 'de-en_test.tsv'), 'r')
annotated_file = open(os.path.join('translation_data', 'annotated_de-en_test.tsv'), 'w')
count = 0
total = 0

lines = original_file.readlines()
for line in tqdm(lines):
    if line.strip() != '' and len(line.strip().split('\t')) == 2:
        de, en = line.strip().split('\t')
        doc_de = nlp_de(de)
        doc_en = nlp_en(en)

        de_label_count = dict()
        en_label_count = dict()
        for ent in doc_de.ents:
            if ent.label_ not in de_label_count:
                de_label_count[ent.label_] = 1
            else:
                de_label_count[ent.label_] += 1
        for ent in doc_en.ents:
            if ent.label_ not in en_label_count:
                en_label_count[ent.label_] = 1
            else:
                en_label_count[ent.label_] += 1

        matching = dict()
        
        for de_ent in doc_de.ents:
            label = de_ent.label_
            if de_label_count[label] == 1 and label in en_label_count and en_label_count[label] == 1:
                for en_ent in doc_en.ents:
                    if en_ent.label_ == label:
                        matching[de_ent] = en_ent
        
        if len(matching) > 0:
            count += 1
        total += 1

        annotated_de = de
        for de_ent, en_ent in matching.items():
            annotated_de = annotated_de.replace(de_ent.text, de_ent.text + f' <{en_ent.text}>')
        
        annotated_file.write(annotated_de + '\t' + en + '\n')
print(count, total)