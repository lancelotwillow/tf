import tensorflow as tf
import numpy as np
from tensorflow import keras
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
word_mapping = dict([(key, value) for (key, value) in word_index.items()])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
num_size = 10000

def encode_review(text):
    import re
    l = re.split(', | ', text)
    return list(map(lambda x: word_mapping.get(x) if word_mapping.get(x) is not None else 2, l))

def classify(review):
	print(review)
	model = keras.models.load_model('./my_model.h5')
	review = keras.preprocessing.sequence.pad_sequences([encode_review(review)],
	                                            value=word_index["<PAD>"],
	                                            padding='post',
	                                            maxlen=256)
	result = model.predict(review)[0]
	if result > 0.6:
		return ('positive', result)
	else:
		return ('negative', result)
