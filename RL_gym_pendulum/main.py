from game_functions import (create_environment,
                            some_random_games,
                            initial_population,
                            load_data,
                            play_the_game)

from rl_model_functions import (load_model,
                                train_model,
                                save_model)

import numpy as np


LR = 1e-3
DATA_DIR = '../.datasets/gym_AI/pendulum/'

GOAL_STEPS = 500
INITIAL_GAMES = 10000
MIN_SCORE = -1100

env = create_environment('Pendulum-v0')

#some_random_games(env, GOAL_STEPS)


training_data = load_data(data_dir=DATA_DIR)
#training
# _data = load_data(data_dir=DATA_DIR, file_name='game_data%d.npy'%MIN_SCORE)


try: 
    print(training_data.shape)
except AttributeError:
    training_data = initial_population(env, INITIAL_GAMES, GOAL_STEPS, DATA_DIR, min_score_requiremenent=MIN_SCORE)

"""
print('X, max {}, min {}, avg {}'.format(np.amax(X), 
                                         np.amin(X), 
                                         np.average(X)))

print('Y, max {}, min {}, avg {}'.format(np.amax(Y), 
                                         np.amin(Y), 
                                         np.average(Y)))
"""
# normalization
#X = (X + 8) / 16
#Y = (Y + 2) / 4

"""
print('X, max {}, min {}, avg {}'.format(np.amax(X), 
                                         1p.amin(X), 
                                         np.average(X)))

print('Y, max {}, min {}, avg {}'.format(np.amax(Y), 
                                         np.amin(Y), 
                                         np.average(Y)))

"""


#print('X', X.shape, X[:2])


for _ in range(10):
	try:
		print(training_data.shape, playing_data.shape)
		train_data = np.append(training_data, playing_data, axis=0)
	except NameError: 
		train_data = training_data

	X = np.array([i[0] for i in train_data]).reshape(-1,len(train_data[0][0]))
	Y = np.array([i[1] for i in train_data])	

	X = (X + 8) / 16
	Y = (Y + 2) / 4

	model = load_model()#model_name='model%d' %MIN_SCORE)
	
	model = train_model(X, Y, model=model, epochs=1)

	save_model(model)#, model_name='model%d' %MIN_SCORE)
	
	n_of_games = 200

	playing_data = play_the_game(env, model, GOAL_STEPS, DATA_DIR, n_of_games=n_of_games, render=False, min_score_requiremenent=MIN_SCORE)


play_the_game(env, model, GOAL_STEPS, DATA_DIR, n_of_games=5, render=True, min_score_requiremenent=MIN_SCORE)