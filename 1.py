# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:16:50 2018

@author: prasann
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.mnist

#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]), len(train_data[5]))
# A dictionary mapping words to an integer index
"""
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<ABC>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[5]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

"""

print(len(train_data[0]), len(test_data[0]))

print(train_data[0])

#print(decode_review(train_data[0]))

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 5000
"""
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='softmax'))
"""
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling2D())
#model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

"""
data = np.random.random((1000,32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_data1 = np.random.random((200, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))    # Model is trained.

print(model.evaluate(val_data, val_labels, batch_size=32))

#model.evaluate(val_data, steps=30)
print(model.predict(val_data1, batch_size=32))

#model.predict(val_data, steps=30)
"""
x_val = train_data[:5000]
partial_x_train = train_data[5000:]

y_val = train_labels[:5000]
partial_y_train = train_labels[5000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=50,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()
#dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


