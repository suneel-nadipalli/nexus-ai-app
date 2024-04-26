import pickle

from keras.preprocessing.sequence import pad_sequences

import pickle

from keras.models import load_model

import numpy as np


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
    
    