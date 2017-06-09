import numpy as np
import math
import matplotlib.pyplot as plt


import keras

from keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import backend as K

DATA_DIR = '../.datasets/sintetic_STS/rot_polinomials/'
MODEL_DIR = 'model/rot_polinomials/'


"""
	LOAD DATA
"""
train_data_in = np.load('%strain_data_in.npy' %DATA_DIR)
val_data_in = np.load('%sval_data_in.npy' %DATA_DIR)
test_data_in = np.load('%stest_data_in.npy' %DATA_DIR)

train_data_out = np.load('%strain_data_out.npy' %DATA_DIR)
val_data_out = np.load('%sval_data_out.npy' %DATA_DIR)
test_data_out = np.load('%stest_data_out.npy' %DATA_DIR)



def seqToSeq_model(n_of_features, 
                   input_sequence_length, 
                   output_sequence_length, 
                   hidden_neurons = 2000, 
                   keep_proc=0.5):
    
    model = keras.models.Sequential()

    ##Encoder
    model.add(LSTM(hidden_neurons,
              input_shape=(input_sequence_length, n_of_features),
              implementation=2,
              return_sequences=False,
              ))
    
    print(model.outputs)
    
    model.add(Dense(hidden_neurons, activation='relu'))
    
    model.add(Dropout(keep_proc))

    print(model.outputs)
    model.add(RepeatVector(output_sequence_length))
    print(model.outputs)
    
    ##Decoder
    model.add(LSTM(hidden_neurons,
                   implementation=2,
                   return_sequences=True))

    print(model.outputs)
    model.add(TimeDistributed(Dense(n_of_features, activation='relu')))
    print(model.outputs)

    return model


def load_model(model_dir):
    """
    This method loads a previously traned model from model_dir.
    Parameters:
        - model_dir (string), a directory where the model and its weights are saved.
    """
    try:
        f = open('%smodel' %model_dir, 'r')
        json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights('%sweights' %model_dir)
        print('Model loaded!')

        return model

    except FileNotFoundError:
        print('Could not load model!')


def save_model(model_dir, model):
    """
    This method saves the model to the model_dir directory.
    Properties:
        - model (keras.model), a trained keras model,
        - model_dir (string), the directory to save the model to.
    """
    json_string = model.to_json()

    f = open('%smodel' %model_dir, 'w+')
    f.write(json_string)
    f.close()

    model.save_weights('%sweights' %model_dir)
    print("Model saved!")


def print_examples(pred, name=None):
    
    print(name)
    a = b = 5
    f, axarr = plt.subplots(a, b)
    #axarr.set_title(name)
    
    for k in range(a):
        for m in range(b):
            i = np.random.randint(val_data_in.shape[0])
            axarr[k][m].scatter(test_data_in[i,:,0],test_data_in[i,:,1], color='blue')
            axarr[k][m].scatter(test_data_out[i,:,0],test_data_out[i,:,1], color='red')
            axarr[k][m].scatter(pred[i,:,0], pred[i,:,1], color='green')

    plt.show()



def main():
    k_p = [0.6] #0.75, 1] #0.25,
    h_n = [1000]

    TRAIN = True
    PREDICT = True

    if TRAIN:
        for kp in k_p:
            for hn in h_n:
                if kp == 0.5 and hn == 1000:
                    continue
                bs = 100
                counter = 0
                
                hparam = 'model_kp_%d_hn_%d/' %(int(kp*100), int(hn/100))
                
                print(hparam)
                
                model = load_model(MODEL_DIR + hparam)

                if model == None:
                    model = seqToSeq_model(2, 15, 5, hidden_neurons=hn, keep_proc=kp)
                
                model.compile(loss='mean_squared_error', optimizer='Adam')
                
                model.fit(train_data_in, 
                          train_data_out, 
                          epochs=2, 
                          batch_size=bs, 
                          validation_data=(val_data_in, val_data_out),
                          callbacks=[TensorBoard(log_dir='log/%s' %hparam, 
                                                 histogram_freq=5, 
                                                 write_graph=False)]
                          )
                

                """
                if K.backend() == 'tensorflow':
                    K.clear_session()
                """
            

    if PREDICT:
        hparams = ['model_kp_60_hn_10/'] # model_kp_75_hn_10/ - This model is quite good!
        #          'model_kp_100_hn_10/',
        #          'model_kp_25_hn_10/', 
        #          'model_kp_50_hn_10/', 
        #          'model_kp_50_hn_15/',

        for hparam in hparams:

            if not TRAIN:
                model = load_model(MODEL_DIR + hparam)
                model.compile(loss='mean_squared_error', optimizer='Adam')

            pred = model.predict(test_data_in)
            print_examples(pred, name=hparam)

    save_model(MODEL_DIR + hparam, model)
    



if __name__ == "__main__":
    main()