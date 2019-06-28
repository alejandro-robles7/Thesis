import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, Dropout, BatchNormalization
from keras.models import Model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.losses import *
from nltk import download

download('stopwords')


def get_embedding_indices(path):
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def get_embeddings(dataframe, labels_index, text_col='Text', label_col='Category'):
    # second, prepare text samples and their labels
    print('Processing text dataset')
    texts = []  # list of text samples
    labels = []  # list of label ids

    for index, row in dataframe.iterrows():
        texts.append(row['Text'])
        labels.append(labels_index[row['Category']])
    print('Found %s texts.' % len(texts))

    return texts, labels

