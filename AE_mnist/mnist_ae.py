import keras 
from keras.layers import Reshape, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.models import model_from_json

import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def toOneHot(array, n_classes):
    # Transforms an array of labels (dim: (-1,)) to a 1-hot array of labels (dim: (-1, n_classes))
    temp_arr = np.zeros((len(array), n_classes))
    temp_arr[np.arange(len(array)), array] = 1
    
    return temp_arr

def getLayerOutput(model, data=None):
    # Function that returns the outputs of each layer in the model.
    from keras import backend as K

    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
    
    if data is None:
        data = np.random.rand(model.input_shape)

    layer_outs = functor([data, 1])
    
    return layer_outs

# The log and model directory
LOGDIR = 'log/1/'
MODELDIR = 'model/1/'

# Model definition: 
model = keras.models.Sequential()

# Encoder:
model.add(Reshape((784,), input_shape=(28,28)))

model.add(Dense(2048, activation="relu"))

model.add(Dropout(0.7))

model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax", name="encoder"))

# Decoder:
model.add(Dense(512, activation="relu"))

model.add(Dense(2048, activation="relu"))

model.add(Dropout(0.7))

model.add(Dense(784, activation="softmax", name="decoder"))

model.add(Reshape([28,28]))

# Tries to load a previously saved model:
try:

        f = open(MODELDIR + 'model', 'r')
        json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights(MODELDIR + 'weights')
        print('Model loaded!')

except FileNotFoundError:
        print('Could not load model!')

# Initializes the model:
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy)

# Trains the model:
model.fit(x=x_train,
          y=x_train,
          batch_size=128, 
          epochs=1000,
          validation_data=(x_test, x_test),
          shuffle=True,
          callbacks=[TensorBoard(log_dir=LOGDIR, 
                                 histogram_freq=100, 
                                 write_graph=True)],
          verbose=1)
"""
# External model testing and evaluation:

y_test = toOneHot(y_train, 10)
y_pred = np.array(getLayerOutput(model, data=x_train)[1])
print(y_test.shape)
print(y_pred)
print(np.sum((np.all(np.equal(y_test, y_pred), axis=1)).astype(int)))
"""
# Saves the model (shape and trained weights):
json_string = model.to_json()

f = open(MODELDIR + 'model', 'w+')
f.write(json_string)
f.close()

model.save_weights(MODELDIR + 'weights')
print("Model saved!")