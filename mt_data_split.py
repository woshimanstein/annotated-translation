import os
import numpy as np

with open(os.path.join('translation_data', 'news-commentary-v14.de-en.tsv'), 'r') as data_file:
    lines = data_file.read()
    documents = []
    sentences = []
    for line in lines.split('\n'):
        if line.strip() == '':
            documents.append(sentences)
            sentences = []
        else:
            sentences.append(line)

    shuffle_indices = np.random.choice(len(documents), len(documents), replace=False)
    idx = 0
    for split, percentage in zip(['train', 'dev', 'test'], [0.98, 0.99, 1]):
        with open(os.path.join('translation_data', f'de-en_{split}.tsv'), 'w') as w:
            while idx < percentage * len(documents):
                for sentence in documents[shuffle_indices[idx]]:
                    w.write(sentence + '\n')
                w.write('\n')
                idx += 1