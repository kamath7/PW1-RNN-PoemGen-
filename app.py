import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # because my GPU sucks

# Text Pre-processing
fileloc = tf.keras.utils.get_file(
    "shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(fileloc, 'rb').read().decode(encoding='utf-8').lower()
# selecting just one part of the text instead of the entire ds
text = text[100000:900000]

characters = sorted(set(text))  # getting unique characters
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))
print(char_to_index, index_to_char)

SEQ_len = 40
step_size = 3

sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_len, step_size):
    sentences.append(text[i: i+SEQ_len])
    next_chars.append(text[i+SEQ_len])

x = np.zeros((len(sentences), SEQ_len,
              len(characters)), dtype=np.bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Model creation

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_len, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)

model.save("textgen.model")
