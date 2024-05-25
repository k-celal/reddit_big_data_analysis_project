import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding,CuDNNGRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
def SentimentAnalysis(text):
    dataset = pd.read_csv('hepsiburada.csv')
    target = dataset['Rating'].values.tolist()
    data = dataset['Review'].values.tolist()
    cutoff = int(len(data) * 0.80)
    x_train, x_test = data[:cutoff], data[cutoff:]
    y_train, y_test = target[:cutoff], target[cutoff:]
    num_words = 10000
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(data)
    x_train_tokens = tokenizer.texts_to_sequences(x_train)
    x_test_tokens = tokenizer.texts_to_sequences(x_test)
    num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
    num_tokens = np.array(num_tokens)
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    loaded_model = load_model('sentiment_analysisModel(99).h5')
    denemeYorumu=["güzel"]
    denemeYorumu.append(text)
    token = tokenizer.texts_to_sequences(denemeYorumu)
    token_pad = pad_sequences(token, maxlen=max_tokens)
    tahmin = loaded_model.predict(token_pad)
    if tahmin[1] > 0.5:
        yorum_tipi = 1
    else:
        yorum_tipi = 0
    return yorum_tipi
    
