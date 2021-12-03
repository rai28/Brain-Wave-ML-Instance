import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# tensorflow keras layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense,Dropout,LSTM,Bidirectional,GRU,SpatialDropout1D,MaxPooling1D
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

def runmodel(string):
    predictor = load_model('ML_Model/CNN_with_LSTM_with_word2vec.h5')
    num_classes = 5
    max_seq_len = 500
    class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
    message = []
    message.append(string)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(message)
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    pred = predictor.predict(padded)
    return class_names[np.argmax(pred)]
