import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, Dropout, BatchNormalization
from keras.models import Model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from zipf import Zipf, getData
import matplotlib.pyplot as plt
from nltk import download

download('stopwords')

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

def get_data(q, path, test_size=0.3, random_state=42):
    df = getData(path)
    z = Zipf(q=q)
    z.fit(df)
    z.transform()
    z.filter_by_language()
    X, y = z['Text'], z['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, z.dataframe


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

    return data, labels

def get_splits(data, labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return x_train, y_train, x_val, y_val

def get_embedding_matrix(word_index, embeddings_index):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def generate_model(num_words, embedding_matrix, labels_index, embedding_dim, max_sequence_length, dropout=0.30):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    print('Training model.')
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
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
                  optimizer='rmsprop',
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

if __name__ == '__main__':

    q = 3
    json_path = 'scrapedsites.json'
    path = 'glove.6B.300d.txt'
    embedding_index = get_embedding_indices(path)
    X_train, X_test, y_train, y_test, full_df = get_data(q, json_path)
    texts, labels = get_embeddings(dataframe=full_df, labels_index=embedding_index)
    data, labels = get_data_and_labels(texts, labels, max_num_words=MAX_NUM_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH)

