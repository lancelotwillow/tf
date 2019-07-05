import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd

num_size = 10000
imdb = keras.datasets.imdb
(train_data, train_label), (test_data, test_label) = \
    imdb.load_data(num_words = num_size)

word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
word_mapping = dict([(key, value) for (key, value) in word_index.items()])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
def encode_review(text):
    import re
    l = re.split(', | ', text)
    return list(map(lambda x: word_mapping.get(x) if word_mapping.get(x) is not None else 2, l))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
model = keras.Sequential()
model.add(keras.layers.Embedding(num_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
#save model
import os
checkpoint_path = "./model_cp"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)
model.fit(train_data,
          train_label,
          epochs=40,
          batch_size=512,
          verbose=1,
          callbacks=[cp_callback])

