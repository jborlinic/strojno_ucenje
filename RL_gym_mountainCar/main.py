from game_functions import (create_environment,
                            some_random_games,
                            initial_population,
                            load_data,
                            play_the_game)

from rl_model_functions import (load_model,
                                train_model,
                                save_model)

import numpy as np
import matplotlib.pyplot as plt


LR = 1e-3
DATA_DIR = '../.datasets/gym_AI/mountainCar/'

GOAL_STEPS = 500
INITIAL_GAMES = 10000

env = create_environment('MountainCar-v0')


training_data = load_data(data_dir=DATA_DIR)

try:
    print(training_data.shape)
except AttributeError:
    training_data = initial_population(env, INITIAL_GAMES, GOAL_STEPS, DATA_DIR)

X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
Y = np.array([i[1] for i in training_data])

print('X', X.shape, X[:2])

model = load_model()

model = train_model(X, Y, model, epochs=10)

save_model(model)

n_of_games = 10
#some_random_games(env, GOAL_STEPS)


play_the_game(env, model, GOAL_STEPS, DATA_DIR, n_of_games=n_of_games, render=True)