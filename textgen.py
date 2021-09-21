# Taken from keras documentations
import random
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # because my GPU sucks

model = tf.keras.models.load_model('textgen.model')
fileloc = tf.keras.utils.get_file(
    "shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(fileloc, 'rb').read().decode(encoding='utf-8').lower()
# selecting just one part of the text instead of the entire ds
text = text[100000:900000]

characters = sorted(set(text))  # getting unique characters
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))


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

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_len - 1)
    generated = ''
    sentence = text[start_index:start_index + SEQ_len]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_len, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated


print(generate_text(300, 0.3))
