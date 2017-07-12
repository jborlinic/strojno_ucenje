import keras
import numpy as np 

from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Activation, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def one_hot(obj, n_classes):
    obj_len = len(obj)
    a = np.array(obj)
    b = np.zeros((obj_len, n_classes))
    b[np.arange(obj_len), a] = 1

    return b

"""
Definicija modela, vsak sloj posebej.
Sequential tip modela predstavlja model, 
ki se izvaja linearno po svojih slojih.
"""
model = keras.models.Sequential()
### Prvi del modela - KONVOLUCIJA
# osnovno preoblikovanje nabora podatkov
model.add(Reshape((28,28,1), input_shape=(28,28)))
# prvi konvolucijski sloj s 32 jedri
model.add(Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
# prvo maksimalno združevanje
model.add(MaxPooling2D(pool_size=2, strides=2))
# drugi konvolucijski sloj s 64 jedri
model.add(Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
# drugo maksimalno združevanje
model.add(MaxPooling2D(pool_size=2, strides=2))

# preoblikovanje matrike iz 4-D v 2-D iz (7,7,64) v (3163,) = 7x7x64
model.add(Flatten())

### Drugi del modela - KLASIFIKATOR
# osnovni (vmesni) polno-povezan sloj s relu aktivacijo
model.add(Dense(units=1024, activation='relu'))
# osipni sloj p = 0.5
model.add(Dropout(0.5))
# zadnji polno-povezan sloj s softmax aktivacijo 
model.add(Dense(units=10, activation='softmax'))


# Inicializacija modela in nastavitev:
#     - kriterijske funkcije,
#     - optimizatorja,
#     - metrike.
model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=['accuracy'])

# Pretvorba razrednih vektorjev
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# Učenje modela, 10 epik (št._epik x št._vseh_podatkov / batch_size)
model.fit(x_train,
          y_train, 
          epochs=10, 
          batch_size=100, 
          verbose=1, 
          callbacks=[TensorBoard(log_dir='log/', histogram_freq=5, write_graph=True)]
)