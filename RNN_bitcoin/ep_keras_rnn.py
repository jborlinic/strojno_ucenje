import keras
from keras.layers import LSTM, Dense, Reshape, Dropout
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras import backend as K

import numpy as np

def split_data(data, val_size=0.2, test_size=0.2):
    """ 
    Splits the data into 3 datasets (train, validation and test), 
    according to the val_size and test_size parameters.
    
    Inputs: 
        - raw_data should be a 1-D array of any length containing sequential data
        - val_size a float32 value representing the percentage of the dataset that should be used for validation data
        - test_size a float32 value representing the percentage of the dataset that should be used for test data
    """
    n_test = int(round(len(data) * (1 - test_size)))
    n_val = int(round(len(data[:n_test]) * (1 - val_size)))

    df_train, df_val, df_test = data[:n_val], data[n_val: n_test], data[n_test:]

    return df_train, df_val, df_test


def rnn_data(data, time_steps):
    """ 
    Transforms the raw data to a model friendly shape consisting of:
        - dataset x (train data), these are vectors of length "time_steps",
        - dataset y (results, values we want to predict).
    
    Inputs:
    data: a sequence of values (1 dimensional array of any length
    time_steps: the number of time steps in the model
    """ 
    return_x = []
    return_y = []
    for i in range((len(data) - time_steps - 1)):
        return_x.append(data[i : i + time_steps])
        return_y.append(data[i + time_steps])

    return np.array(return_x), np.array(return_y)

def test_layers(model):
    """
    Function that returns outputs of each individual layer in the model.
    
    Inputs: the model
    """
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

    # Testing
    test = np.random.random(input_shape)[np.newaxis,...]
    layer_outs = functor([test, 1.])
    print(layer_outs)


def relu_act(x):
    # the actiavtion function of the final layer in the model, with the maximall bound at 1
    return keras.activations.relu(x, max_value=1)


"""
The main function containing:
    - data preperation,
    - model definition,
    - model saving,
    - model initialization,
    - model training (fitting to the data),
    - model evaluation.
    
Inputs:
    raw_data should be a 1-D array of any length, containing sequential data
"""
def BTC_pred_rnn(raw_data, 
                 logdir='log/', 
                 training_steps= 10000, 
                 lstm_depth=5, 
                 time_step=10,
                 batch_size=128,
                 model_dir='model/unnamed/'):
    
    
    # Data preperation:
    train_data, val_data, test_data = split_data(raw_data, 
                                                 val_size=0.15, 
                                                 test_size=0)
    
    
    train_x, train_y = rnn_data(train_data, time_step)
    val_x, val_y = rnn_data(val_data, time_step)
    
    # Model definition:
    model = keras.models.Sequential()
    
    model.add(Reshape((10,1), input_shape=(10,)))

    model.add(LSTM(time_step,
              activation='tanh',
              recurrent_activation='hard_sigmoid', 
              use_bias=True, 
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              unit_forget_bias=True
              ))

    model.add(Dense(10))

    model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(Dense(1, activation=relu_act))





    # Model initialization:
    model.compile(loss="mean_absolute_percentage_error",
                  optimizer="Adagrad",
                  metrics=['accuracy'])
    
    # Model training:
    model.fit(train_x,
                train_y, 
                epochs=100,
                batch_size=100, 
                verbose=1, 
                validation_data=(val_x, val_y),
                callbacks=[TensorBoard(log_dir='log/', 
                                       histogram_freq=100, 
                 
                                       write_graph=True)]
                )
    
    # Testing layer outputs and model evaluation:
    test_layers(model)
    
    print('Predicting:')
    print(train_x[56:71,9])
    print(model.predict(train_x[55:70],
                        batch_size=100,
                        verbose=2).T)
    
    # Saving the model:
    json_string = model.to_json()
    
    f = open(model_dir + 'model', 'w+')
    f.write(json_string)
    f.close()

    model.save_weights(model_dir + 'weights')
    print("Model saved!")