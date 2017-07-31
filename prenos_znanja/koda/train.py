import keras

from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout
from keras.callbacks import TensorBoard

from support_functions import one_hot, save_model

MODEL_DIR = 'model/'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

"""
Definicija modela po slojih.
Sequential tip modela predstavlja model, ki se izvaja linearno po svojih slojih.
"""
model = keras.models.Sequential()

### Prvi del modela - KONVOLUCIJA

# osnovno preoblikovanje nabora podatkov - potrebno zaradi oblike vhodnih podatkov pri naslednjem sloju (Conv2D)
model.add(Reshape((28,28,1), input_shape=(28,28)))
# (trenutna oblika podatkov (28,28,1))

# prvi konvolucijski sloj s 32 jedri 
model.add(Conv2D(8, kernel_size=5, strides=1, padding='same', activation='relu'))
# (trenutna oblika podatkov (28,28,8))

# prvo maksimalno združevanje 
model.add(MaxPooling2D(pool_size=2, strides=2))
# (trenutna oblika podatkov (14,14,8))

# drugi konvolucijski sloj s 64 jedri 
model.add(Conv2D(16, kernel_size=5, strides=1, padding='same', activation='relu'))
#(trenutna oblika podatkov (14,14,16))

# drugo maksimalno združevanje 
model.add(MaxPooling2D(pool_size=2, strides=2))
# (trenutna oblika podatkov (7,7,16))

# preoblikovanje matrike iz 4-D v 2-D iz (7,7,16) v (784) = 7x7x16
model.add(Flatten())
# (trenutna oblika podatkov (784))

### Drugi del modela - KLASIFIKATOR

# osnovni (vmesni) polno-povezan sloj s RwLU aktivacijo
model.add(Dense(units=300, activation='relu'))
# (trenutna oblika podatkov (300))

# osipni sloj p = 0.5
model.add(Dropout(0.5))
# (trenutna oblika podatkov (300))

# zadnji polno-povezan sloj s softmax aktivacijo 
model.add(Dense(units=10, activation='softmax'))
# (trenutna oblika podatkov (10))

# Inicializacija modela in nastavitev:
#     - kriterijske funkcije,
#     - optimizatorja,
#     - metrike.
model.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=['accuracy'])

# Pretvorba razrednih vektorjev iz (št_primerov, 1) v (št_primerov, 10)
# "one hot" vektor je binarni vektor z eno vrednostjo 1 (npr. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 
# ki nam pove v kateri razred spada trenutna slika (primer: 4)
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# Učenje modela, 2 epohi 
# dejansko št. ponovitev = št._epoh x št._vseh_podatkov / batch_size
model.fit(x_train,
          y_train, 
          epochs=2, 
          batch_size=100, 
          verbose=1, 
          callbacks=[TensorBoard(log_dir='log/', write_graph=True)]
)

# metoda shrani strukturo modela - model in naučene uteži v direktorij MODEL_DIR
save_model(MODEL_DIR, model)