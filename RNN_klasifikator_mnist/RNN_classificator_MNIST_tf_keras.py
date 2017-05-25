import tensorflow as tf
import numpy as np
keras = tf.contrib.keras

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
Define the model.
"""

l_len = len(y_train)
b = np.zeros((l_len, 10))
b[np.arange(l_len), y_train] = 1
y_train = b


l_len = len(y_test)
b = np.zeros((l_len, 10))
b[np.arange(l_len), y_test] = 1
y_test = b

x = keras.layers.Input(shape=(28,28))

lstm = keras.layers.LSTM(256)(x)

pred = keras.layers.Dense(10, activation="softmax")(lstm)

model = keras.models.Model(x, pred)

model.compile(optimizer=keras.optimizers.Adam(),
			  loss=keras.losses.categorical_crossentropy)

TensorBoard = keras.callbacks.TensorBoard(log_dir=LOGDIR+'tb/', histogram_freq=0, write_graph=True)

model.fit(x=x_train,
		  y=y_train,
		  batch_size=100, 
		  epochs=1, 
		  validation_data=(x_test, y_test),
		  shuffle=True,
		  verbose=1,
		  callbacks=TensorBoard)


keras.models.save_model(model, LOGDIR + 'model.hdf5', overwrite=True)