import sys
import argparse
from io import BufferedReader, BufferedReader

import nltk
import tqdm
from readchar import readkey

american_english_dict = "/usr/share/dict/american-english"
british_english_dict = "/usr/share/dict/british-english"
translation_data = "translation_data/de-en_train.tsv"

def load_english_dict_no_name():
    english_dict = set()
    with open(american_english_dict, "r") as f:
        for word in f:
            if not is_capitalized(word):
                english_dict.add(word.strip())
    with open(british_english_dict, "r") as f:
        for word in f:
            if not is_capitalized(word):
                english_dict.add(word.strip())
    return english_dict

def is_capitalized(string):
    return string[0].isupper()

def key_yes_no():
    while True:
        key = readkey()
        if key == "y":
            return True
        elif key == "n":
            return False
        elif key == "\x03":
            raise RuntimeError("Escaped")

def main():
    english_dict = load_english_dict_no_name()
    name_set = set()
    not_name_set = set()
    already_match_lines = []
    with open(translation_data, "r") as f:
        lines = f.readlines()
        for i in tqdm.trange(len(lines)):
            line = lines[i]
            split_line = line.split("\t")
            if len(split_line) != 2:
                continue
            replace = set()
            de, en = split_line
            printed = False
            split_en = nltk.word_tokenize(en, language='english')
            split_de = nltk.word_tokenize(de, language='german')
            match = True
            for j, word in enumerate(split_en):
                if is_capitalized(word):
                    if word.lower() in english_dict:
                        continue
                    elif word in split_de:
                        replace.add(word)
                        continue
                    else:
                        match = False
                        break
                    #elif word in not_name_set:
                    #    continue
                    #print(split_en)
                    #print(word, i, len(lines))
                    #try:
                    #    if word in name_set:
                    #        match = False
                    #        break
                    #    elif not key_yes_no():
                    #        not_name_set.add(word)
                    #    else:
                    #        name_set.add(word)
                    #        match = False
                    #        break
                    #except:
                    #    for l in already_match_lines:
                    #        print(l)
                    #    print(name_set)
                    #    print(not_name_set)
                    #    return
                        
            if match:
                for word in replace:
                    de = de.replace(word, "{} <{}>".format(word, word))
                already_match_lines += ["{}\t{}".format(de, en)]
    with open("annotated_data.tsv", "w") as f:
        f.writelines(already_match_lines)

if __name__ == "__main__":
    main()