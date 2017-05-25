"""
This is the file to call from the comandline (python3 main.py).
It contains the hyperparameters for the BTC_pred_rnn model and the reading or creating of the dataset used for the training of the model.
"""

from ep_keras_rnn import BTC_pred_rnn
import os
import numpy as np 

# The function checks if all the directories used in the model exist and creates them if they don't.
def check_directories(list_of_dirs):
	for directory in list_of_dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)

# The function creates a dataset of values that correspond to the values of the sin function.
def sin_wave(n):
	f = np.arange(n)
	f = f.astype('float32')
	f = np.sin(f) * 10 + 10
	return f

# Hyperparameters for the BTC_pred_model
MODEL_NAME = 'test'                # string - the name of the model and directories
TRAINING_STEPS = 10000             # integer - number of training steps
LSTM_DEPTH = 5                     # integer - depth of the LSTM model (number of cells)
TIME_STEP = 10                     # integer - length of the input vectors
BATCH_SIZE = 128                   # integer - size of the batch
DATA_DIR = '../.datasets/bitcoin/' # string - directory where the data is stored


MODEL_DIR = 'model/%s/' %MODEL_NAME
LOG_DIR = 'log/%s/' %MODEL_NAME

"""
Different datasets:
data.npy: timestamp;value;trade_size
bitstampUSD.csv: timestamp;value;trade_size
sample_data.npy: timestamp;value;trade_size

ohlc.npy: open;high;low;close
bitstampUSD_chopsticks.csv: timestamp,open,high,low,close,volume(BTC),volume(USD),weighted_price
"""

check_directories([MODEL_DIR, LOG_DIR])

data = np.load('%sohlc.npy' %DATA_DIR)
data = data[:,3]
data = data / 100

#data = np.arange(3000)%20
#data = sin_wave(5000)


print(data[55:70])

BTC_pred_rnn(data,#[:,0],
				 logdir=LOG_DIR, 
				 training_steps=TRAINING_STEPS, 
				 lstm_depth=LSTM_DEPTH, 
				 time_step=TIME_STEP,
				 batch_size=BATCH_SIZE,
				 model_dir=MODEL_DIR)