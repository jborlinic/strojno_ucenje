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
	

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

model = keras.models.Sequential()

model.add(Reshape((28,28,1), input_shape=(28,28)))

model.add(Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

model.add(Dense(units=1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss="categorical_crossentropy",
			  optimizer="Adadelta",
			  metrics=['accuracy'])

model.fit(x_train,
		  y_train, 
		  epochs=10, 
		  batch_size=100, 
		  verbose=1, 
		  validation_data=(x_test, y_test),
		  callbacks=[TensorBoard(log_dir='log/', histogram_freq=5, write_graph=True)]
)



