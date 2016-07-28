#!/usr/bin/env python
"""Data loading and processing"""

# python imports
import os, sys, codecs
from collections import OrderedDict

def load_data(data_dir):
    """Load the data from the data dir and return sequences of texts and labels"""
    texts, labels = [], []
    for filename in os.listdir(data_dir):
        with codecs.open(os.path.join(data_dir, filename), "r", encoding = "utf-8") as fp:
            data = [x.strip("\n") for x in fp.readlines()]
            text_sequence = [x.split()[0] for x in data if x.split()[1] != 'BB']
            label_sequence = [x.split()[1] for x in data if x.split()[1] != 'BB']
            texts.append(text_sequence)
            labels.append(label_sequence)
    
    return texts, labels

def generate_word2index(texts):
    """Generate word2index mapping"""
    word2index = {}
    word2index['_pad_'] = 0
    for text in texts:
        for words in text:
            if words not in word2index:
                word2index[words] = len(word2index)

    return word2index

def process_data(data_dir):
    """Process the data and return sequences of texts and labels in terms of sequences of indices"""

    # labels2index mapping
    labels2index = {"B" : 0, "NB" : 1}
    # Generating sequences of texts and labels from the data
    text, labels = load_data(data_dir)
    # Generating word2index mapping
    words2index = generate_word2index(text)
    #Converting sequences of texts / labels to sequences of indices
    text_sequences = [[words2index[words] for words in sentence] for sentence in text]
    label_sequences = [[labels2index[label] for label in sequence] for sequence in labels]

    return text_sequences, label_sequences, words2index, labels2index
