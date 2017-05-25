import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from keras.models import model_from_json

MODEL_DIR = 'model/'
DATA_DIR = '../.datasets/gps_tracks/'


def load_data():
	train_data_in = np.load('%strain_data_in.npy' %DATA_DIR)
	val_data_in = np.load('%sval_data_in.npy' %DATA_DIR)
	test_data_in = np.load('%stest_data_in.npy' %DATA_DIR)

	train_data_out = np.load('%strain_data_out.npy' %DATA_DIR)
	val_data_out = np.load('%sval_data_out.npy' %DATA_DIR)
	test_data_out = np.load('%stest_data_out.npy' %DATA_DIR)

	return train_data_in, train_data_out, val_data_in, val_data_out, test_data_in, test_data_out

def seqToSeq_model(n_of_features, input_sequence_length, output_sequence_length):
    
    hidden_neurons = 2300
    model = keras.models.Sequential()

    ##Encoder
    model.add(LSTM(hidden_neurons,
              input_shape=(input_sequence_length, n_of_features),
              return_sequences=False
              ))
    print(model.outputs)
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(0.5))

    print(model.outputs)
    model.add(RepeatVector(output_sequence_length))
    print(model.outputs)
    
    ##Decoder
    model.add(LSTM(hidden_neurons,
              return_sequences=True))

    print(model.outputs)
    model.add(TimeDistributed(Dense(n_of_features, activation='relu')))
    print(model.outputs)

    return model

def load_model(model_dir):
    """
    This function loads a previously traned model from model_dir.
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
    This function saves the model to the model_dir directory.
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


def test_output(model, n_of_graphs=50):
	pred = model.predict(test_data_in)
	for _ in range(n_of_graphs):
	    
	    i = np.random.randint(val_data_in.shape[0])
	    print(i)
	    
	    plt.scatter(test_data_in[i,:,0],test_data_in[i,:,1], color='blue')
	    plt.scatter(test_data_out[i,:,0],test_data_out[i,:,1], color='red')
	    plt.scatter(pred[i,:,0], pred[i,:,1], color='green')
	    plt.show()



train_data_in, train_data_out, val_data_in, val_data_out, test_data_in, test_data_out  = load_data()


model = seqToSeq_model(2, 15, 5)

#model = load_model(MODEL_DIR)


model.compile(loss='mean_squared_error', optimizer='Adam')

model.fit(train_data_in, 
		  train_data_out, 
		  epochs=10, 
		  batch_size=100, 
		  validation_data=(val_data_in, val_data_out))

save_model(MODEL_DIR, model)



test_output(model)

