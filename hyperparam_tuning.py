from full_zipf_cnn import *
from keras.optimizers import Adam, RMSprop
import talos as ta
from talos.model.normalizers import lr_normalizer

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
embedding_matrix = None
labels_index = None


def cnn_model(x_train, y_train, x_val, y_val, params):
    embedding_layer = Embedding(MAX_NUM_WORDS,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Training model.')
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu', padding = 'same')(x)
    x = MaxPooling1D(5)(x)
    # x = BatchNormalization()(x)
    x = Conv1D(128, 5, activation='relu', padding = 'same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(params['dropout'])(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=(x_val, y_val))
    return history, model

p = {'lr': (0.001, 0.005, 0.01),
     'batch_size': (32, 64, 128),
     'epochs': [20, 30, 40],
     'dropout': (0.1, 0.2, 0.3),
     'optimizer': [Adam, RMSprop],
     }


t = ta.Scan(x=x,
            y=y,
            model=cnn_model,
            grid_downsample=0.01,
            params=p,
            dataset_name='websites',
            experiment_no='1')