import numpy as np 
import keras
from keras.layers import Dense, LSTM, Embedding, Conv1D, Dropout
from keras.datasets import imdb
from keras.models import Sequential, model_from_json
from keras.preprocessing import sequence

model_dir = 'model/'
max_words = 5000              # the number of different words in the word encoding
max_review_length = 500       # the number of words in each review
embedding_vector_length = 32

# gather the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# pad/cut each review sequence to the same size (max_review_length)
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# create the model
model = Sequential()

# input layer with vector embedding
model.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))

# 1D convolution layer
""" 
The idea behind this layer is to extract extra features from the neighbouring words
it can be tought of as a "phrase detector".
"""
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

# the recurrent LSTM(cell) layer
model.add(LSTM(10))

# "mandatory" dropout layer to prevent overfitting
model.add(Dropout(0.7))

# finnal fully-connected layer for classification
model.add(Dense(1, activation='sigmoid'))

# model loading part, it tries to load a previously saved model
try:

    f = open(model_dir + 'model', 'r')
    json_string = f.read()

    model = model_from_json(json_string)
    model.load_weights(model_dir + 'weights')
    print('Model loaded!')

except FileNotFoundError:
    print('Could not load model!')

# model initialization with:
#  - classic binnary crossentropy loss
#  - well known adam optimizer
#  - accuracy metric (keras will report the accuracy of the model every epoch, this helps with model learning suppervision)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# the function that fits the model to the data (i.e. model learning phase)
model.fit(x_train, 
          y_train, 
          validation_data=(x_test, y_test),
          verbose=1, 
          epochs=6, 
          batch_size=64)

# additional evaluation of the model, after the learning phase
scores = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy: %.2f%%' %(scores[1]*100))

# model saving part, it saves the model weights and structure to two files
json_string = model.to_json()

f = open(model_dir + 'model', 'w+')
f.write(json_string)
f.close()

model.save_weights(model_dir + 'weights')
print("Model saved!")