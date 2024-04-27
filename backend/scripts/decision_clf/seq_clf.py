import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.optimizers import Adamax

from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras.layers import Dense, ReLU

from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout

from tensorflow.keras.models import Sequential, Model

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import pickle

from tensorflow.keras.models import load_model

def prep_data():

    # Assuming df is your DataFrame and you want to split based on 'col' column
    # You can adjust the test_size and val_size to change the split proportions
    train_size = 0.9
    test_size = 0.05
    val_size = 0.05

    df = pd.read_csv('../../data/output/decisions.csv')

    df = df[['text', 'decision']]

    # First split into train and (test + val)
    df, test_val_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)

    # Then split test_val_df into test and validation sets
    test_df, val_df = train_test_split(test_val_df, test_size=val_size/(test_size + val_size), random_state=42)
    
    return df, test_df, val_df

def split_data():

    df, test_df, val_df = prep_data()

    X_train = df['text']
    y_train = df['decision']

    X_test = test_df['text']
    y_test = test_df['decision']

    X_val = val_df['text']
    y_val = val_df['decision']

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    
    y_val = encoder.transform(y_val)
    
    y_test = encoder.transform(y_test)

    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))

    return X_train, y_train, X_test, y_test, X_val, y_val, mapping

def prep_model():

    max_words = 10000
    
    max_len = 50
    
    embedding_dim = 32 

    # Branch 1
    branch1 = Sequential()
    branch1.add(Embedding(max_words, embedding_dim, input_length=max_len))
    branch1.add(Conv1D(64, 3, padding='same', activation='relu'))
    branch1.add(BatchNormalization())
    branch1.add(ReLU())
    branch1.add(Dropout(0.5))
    branch1.add(GlobalMaxPooling1D())

    # Branch 2
    branch2 = Sequential()
    branch2.add(Embedding(max_words, embedding_dim, input_length=max_len))
    branch2.add(Conv1D(64, 3, padding='same', activation='relu'))
    branch2.add(BatchNormalization())
    branch2.add(ReLU())
    branch2.add(Dropout(0.5))
    branch2.add(GlobalMaxPooling1D())

    concatenated = Concatenate()([branch1.output, branch2.output])

    hid_layer = Dense(128, activation='relu')(concatenated)
    dropout = Dropout(0.3)(hid_layer)
    output_layer = Dense(2, activation='softmax')(dropout)

    model = Model(inputs=[branch1.input, branch2.input], outputs=output_layer)

    model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall()])
    
    return model

def train_model():

    X_train, y_train, X_test, y_test, X_val, y_val, mapping = split_data()

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train) 
                                
    sequences = tokenizer.texts_to_sequences(X_train)

    tr_x = pad_sequences(sequences, maxlen=50) 
    tr_y = to_categorical(y_train)

    sequences = tokenizer.texts_to_sequences(X_val)
    val_x = pad_sequences(sequences, maxlen=50)
    val_y = to_categorical(y_val)

    sequences = tokenizer.texts_to_sequences(X_test)
    ts_x = pad_sequences(sequences, maxlen=50)
    ts_y = to_categorical(y_test)

    model = prep_model()

    batch_size = 256
    epochs = 100
    history = model.fit([tr_x, tr_x], tr_y, epochs=epochs, batch_size=batch_size,
                        validation_data=([val_x, val_x], val_y))
    
    
    with open('../../data/models/dec_clf/tokenizer.pkl', 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    model.save('../../data/models/dec_clf/nlp.h5')

def predict(text, model_path, token_path):
    
    model = load_model(model_path)
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    sequences = tokenizer.texts_to_sequences([text])
    x_new = pad_sequences(sequences, maxlen=50)
    predictions = model.predict([x_new, x_new])
    
    mapping = {0: 'no', 1: 'yes'}
    
    probs = list(predictions[0])
    
    max_idx = np.argmax(probs)
    
    return mapping[max_idx]
    
    