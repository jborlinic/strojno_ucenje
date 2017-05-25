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
DATA_DIR = '../.datasets/gym_AI/cartpole/'

GOAL_STEPS = 500
INITIAL_GAMES = 10000
MIN_SCORE = 100

env = create_environment('CartPole-v1')


training_data = load_data(data_dir=DATA_DIR)
#training
# _data = load_data(data_dir=DATA_DIR, file_name='game_data%d.npy'%MIN_SCORE)


try: 
    print(training_data.shape)
except AttributeError:
    training_data = initial_population(env, INITIAL_GAMES, GOAL_STEPS, DATA_DIR)

X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
Y = [i[1] for i in training_data]

#print('X', X.shape, X[:2])

model = load_model()#model_name='model%d' %MIN_SCORE)

#model = train_model(X, Y, model, epochs=10)

#save_model(model)#, model_name='model%d' %MIN_SCORE)
n_of_games = 10

some_random_games(env, GOAL_STEPS)

play_the_game(env, model, GOAL_STEPS, DATA_DIR, n_of_games=n_of_games, render=True)