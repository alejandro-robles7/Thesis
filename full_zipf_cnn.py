import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_pickle, read_csv
from math import pow
from collections import Counter
from json import loads
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import Constant
from nltk import download

download('stopwords')


class Zipf:

    target_column = ''
    vector_column = 'Counter_Text'
    output_column = 'Clean_Text'


    def __init__(self, q=2, lower=True):
        self.q = q
        self.lower = lower

    def fit(self, dataframe, target_column='Text'):
        self.target_column = target_column
        self.dataframe = dataframe

    def transform(self, output_column='Clean_Text'):
        self.output_column = output_column
        self._vectorize()
        self._cleanText()



    def _vectorize(self):
        clean_text = []
        if self.lower:
            self.dataframe[self.target_column] = self.dataframe[self.target_column].str.lower()
        for text in self.dataframe[self.target_column]:
            try:
                temp = Zipf.getJ1(text, self.q, True)
            except:
                temp = []
            clean_text.append(temp)
        self.dataframe[self.vector_column] = clean_text


    def _cleanText(self):
        clean_text = []
        for text_vector in self.dataframe[self.vector_column]:
            try:
                words = self.get_strings(text_vector)
                no_stops = self.remove_stopwords(words)
                clean_row = ' '.join(no_stops)
            except:
                clean_row = np.nan
            clean_text.append(clean_row)
        self.dataframe[self.output_column] = clean_text



    @staticmethod
    def getK(q, n=3):
        return (pow(q, n) - 1) / (q - 1)

    @staticmethod
    def getJ(arr, q, n=3):
        c = arr.sum()
        k = Zipf.getK(q, n)
        j2 = c / k
        j1 = j2 * q
        j0 = j1 * q
        return [j0, j1, j2]

    @staticmethod
    def findindex(arr, value):
        return (np.abs(arr.cumsum() - value)).argmin()

    @staticmethod
    def getIndices(arr, j):
        range0 = Zipf.findindex(arr, j[0]) + 1
        range1 = Zipf.findindex(arr[range0 + 1:], j[1]) + 1
        return [(0, range0), (range0, range0 + range1), (range0 + range1, len(arr))]

    @staticmethod
    def getSubset(arr, tup):
        return arr[tup[0]:tup[1]]

    @staticmethod
    def checkCard(arr, indices):
        subs = [Zipf.getSubset(arr, ind) for ind in indices]
        lens = [len(sub) for sub in subs]
        return subs, lens

    @staticmethod
    def getJ1(text, q, split=False):
        words_dict = Zipf.get_counter(text, split)
        word_counts = np.array([word[1] for word in words_dict])
        j = Zipf.getJ(word_counts, q)
        s = Zipf.getIndices(word_counts, j)
        return Zipf.getSubset(words_dict, s[1])

    @staticmethod
    def get_counter(words, split=False):
        if split:
            words = words.split()
        return Counter(words).most_common()

    @staticmethod
    def get_strings(arr, index=0):
        return [w[index] for w in arr]

    @staticmethod
    def remove_stopwords(words):
        return [word for word in words if word not in stopwords.words('english')]






def get_raw_data(path='scrapedsites.json'):
    data = []
    with open(path) as f:
        for line in f:
            d = loads(line)
            data.append(d)

        dataFrame = DataFrame(data)
    return dataFrame

def get_data(path='english_sites.pkl'):
    df = None
    if path.endswith('pkl'):
        df = read_pickle(path)
    elif path.endswith('.json'):
        df = get_raw_data(path)
    elif path.endswith('.txt'):
        df = read_csv(path, encoding='utf-8', header=None, names=['Category', 'Text'], sep=' ')
    return df


def use_Zipf(df, q):
    z = Zipf(q=q)
    z.fit(df)
    z.transform()
    return z.dataframe


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
        texts.append(row[text_col])
        labels.append(labels_index[row[label_col]])
    print('Found %s texts.' % len(texts))
    return texts, labels

def get_data_and_labels(texts, labels, max_num_words, max_sequence_length):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_sequence_length)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return data, labels, word_index

def get_splits(data, labels, validation_split):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(validation_split * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return x_train, y_train, x_val, y_val

def get_embedding_matrix(word_index, embeddings_index, embedding_dim, max_num_words):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(max_num_words, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words

def generate_model(num_words, embedding_matrix, labels_index, embedding_dim, max_sequence_length, dropout=0.30, optimizer='rmsprop'):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    print('Training model.')
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    # x = BatchNormalization()(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    print(model.summary())
    return model

def train_model(model, x_train, y_train, x_val, y_val, batch_size=512, epochs=20):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val))

    # model.save('model_keras3.h5')
    # model.save_weights('model_weights3.h5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.show()
    return history

def find_q():
    # Parameters for model
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2

    # Paths
    data_path = 'english_sites.pkl'
    embeddings_path = 'glove.6B.300d.txt'
    labels_index = {'games & toys' : 0, 'sports' : 1, 'travel' : 2, 'food & drink' : 3}

    # Get English sites
    df = get_data(data_path)

    # best q
    q_best = 0
    accuracy_best = 0

    q_s = [2,3,4,5]
    for q in q_s:
        # Use Zipf's Law to preprocess text
        zipf_data = use_Zipf(df, q)

        # Get Embeddings
        embedding_index = get_embedding_indices(embeddings_path)

        # Get embeddings for our training data
        texts, labels = get_embeddings(dataframe=zipf_data, labels_index=labels_index)

        # Feature Extraction
        data, labels, word_index = get_data_and_labels(texts, labels, max_num_words=MAX_NUM_WORDS,
                                                       max_sequence_length=MAX_SEQUENCE_LENGTH)

        # Split Data
        x_train, y_train, x_val, y_val = get_splits(data, labels, validation_split=VALIDATION_SPLIT)

        # Embedding Matrix
        embedding_matrix, num_words = get_embedding_matrix(word_index, embeddings_index=embedding_index,
                                                           embedding_dim=EMBEDDING_DIM, max_num_words=MAX_NUM_WORDS)

        # Generate Model
        model = generate_model(num_words, embedding_matrix, labels_index, embedding_dim=EMBEDDING_DIM,
                               max_sequence_length=MAX_SEQUENCE_LENGTH, dropout=0.30)

        # Train Model
        history = train_model(model, x_train, y_train, x_val, y_val, batch_size=512, epochs=20)

        # Get Best Accuracy
        accuracy = np.array(history.history['val_acc']).max()

        if accuracy > accuracy_best:
            accuracy_best = accuracy
            q_best = q

    return q_best, accuracy_best

def start_process(df, q, embedding_index, batch_size, epochs, dropout, optimizer):
    zipf_data = use_Zipf(df, q)


    labels_index = {'games & toys' : 0, 'sports' : 1, 'travel' : 2, 'food & drink' : 3}


    # Get embeddings for our training data
    texts, labels = get_embeddings(dataframe=zipf_data, labels_index=labels_index)

    # Feature Extraction
    data, labels, word_index = get_data_and_labels(texts, labels, max_num_words=MAX_NUM_WORDS,
                                                   max_sequence_length=MAX_SEQUENCE_LENGTH)

    # Split Data
    x_train, y_train, x_val, y_val = get_splits(data, labels, validation_split=VALIDATION_SPLIT)

    # Embedding Matrix
    embedding_matrix, num_words = get_embedding_matrix(word_index, embeddings_index=embedding_index,
                                                       embedding_dim=EMBEDDING_DIM, max_num_words=MAX_NUM_WORDS)

    # Generate Model
    model = generate_model(num_words, embedding_matrix, labels_index, embedding_dim=EMBEDDING_DIM,
                           max_sequence_length=MAX_SEQUENCE_LENGTH, dropout=dropout, optimizer=optimizer)

    # Train Model
    history = train_model(model, x_train, y_train, x_val, y_val, batch_size=batch_size, epochs=epochs)

    # Get Best Accuracy
    accuracy = np.array(history.history['val_acc']).max()

    return accuracy

if __name__ == '__main__':

    # Parameters for model
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2

    # Paths
    json_path = 'scrapedsites.json'
    data_path = 'english_sites.pkl'
    embeddings_path = 'glove.6B.300d.txt'

    # Zipf's Law Parameter
    q = 3

    # Get English sites
    df = get_data(data_path)

    # Use Zipf's Law to preprocess text
    zipf_data = use_Zipf(df, q)


    # Get Embeddings
    embedding_index = get_embedding_indices(embeddings_path)

    # Label Index
    labels_index = {'games & toys' : 0, 'sports' : 1, 'travel' : 2, 'food & drink' : 3}

    # Get embeddings for our training data
    texts, labels = get_embeddings(dataframe=zipf_data, labels_index=labels_index)

    # Feature Extraction
    data, labels, word_index = get_data_and_labels(texts, labels, max_num_words=MAX_NUM_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH)

    # Split Data
    x_train, y_train, x_val, y_val = get_splits(data, labels, validation_split=VALIDATION_SPLIT)

    # Embedding Matrix
    embedding_matrix, num_words = get_embedding_matrix(word_index, embeddings_index=embedding_index, embedding_dim=EMBEDDING_DIM, max_num_words=MAX_NUM_WORDS)

    # Generate Model
    model = generate_model(num_words, embedding_matrix, labels_index, embedding_dim=EMBEDDING_DIM, max_sequence_length=MAX_SEQUENCE_LENGTH, dropout=0.30)

    # Train Model
    history = train_model(model, x_train, y_train, x_val, y_val, batch_size=512, epochs=20)


