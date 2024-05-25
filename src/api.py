from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global değişkenler
loaded_model = None
tokenizer = None

def load_data():
    dataset = pd.read_csv('src/hepsiburada.csv')
    target = dataset['Rating'].values.tolist()
    data = dataset['Review'].values.tolist()
    cutoff = int(len(data) * 0.80)
    x_train, x_test = data[:cutoff], data[cutoff:]
    y_train, y_test = target[:cutoff], target[cutoff:]
    return x_train, y_train, x_test, y_test

def load_tokenizer(data):
    num_words = 10000
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(data)
    return tokenizer

def load_model_file():
    return load_model('src/sentiment_analysisModel(99).h5')

def initialize_model():
    global loaded_model, tokenizer
    x_train, y_train, x_test, y_test = load_data()
    tokenizer = load_tokenizer(x_train + x_test)
    loaded_model = load_model_file()

@app.route('/sentiment', methods=['POST'])
def predict_sentiment():
    global loaded_model, tokenizer
    data = request.get_json()
    text = data['text']
    
    if loaded_model is None or tokenizer is None:
        initialize_model()
    
    denemeYorumu = ["güzel", text]
    token = tokenizer.texts_to_sequences(denemeYorumu)
    
    # Hesaplanan maksimum token sayısı
    max_tokens = 59
    
    token_pad = pad_sequences(token, maxlen=max_tokens)
    tahmin = loaded_model.predict(token_pad)
    
    if tahmin[1] > 0.5:
        sentiment = 1
    else:
        sentiment = 0
        
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='172.20.10.13',port='8080',debug=False)
