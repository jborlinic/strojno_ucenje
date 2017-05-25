import gym
import random
import numpy as np
import time

from statistics import mean, median
from collections import Counter

def create_environment(env_name):
    env = gym.make(env_name)
    env.reset()
    return env

def some_random_games(env, goal_steps, n_of_games=5):
    actions = []
    for episode in range(n_of_games):
        env.reset()
        for _ in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            actions.append(action)
            print(action)
            observation, reward, done, info = env.step(action)
            if done: 
                break

    actions = np.array(actions)
    print('Actions, max {}, min {}, avg {}'.format(np.amax(actions), 
                                                   np.amin(actions), 
                                                   np.average(actions)))
    print('%d random games played.' %n_of_games)
    time.sleep(2)


def load_data(file_name='game_data.npy', data_dir='data/'):
    try:
        data = np.load(data_dir + file_name)
        print('Data loaded!')

        return data

    except IOError:
        print('Could not find data file!')
        return None


def save_data(data, file_name='game_data.npy', add_data=False, data_dir='data/'):
    if data.shape[0] > 0:
        if add_data:
            old_data = load_data(file_name=file_name, data_dir=data_dir)
            if old_data.shape[0] > 0:
                try: 
                    print(old_data.shape, data.shape)
                    data = np.concatenate((old_data, data), axis=0)
                except AttributeError:
                    print('No old data found!')
        np.save(data_dir + file_name, data)
        print('Data saved!')


def initial_population(env, n_of_games, goal_steps, data_dir, min_score_requiremenent=50, n_of_moves=2):
    training_data = []
    scores = []
    accepted_scores = []
    print('Generating %d initial population!' %n_of_games)
    for _ in range(n_of_games):
        score = 0 
        game_memmory = []
        prev_observation = []
        
        for _ in range(goal_steps):
            action = [np.random.rand() * 4 - 2]
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memmory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            
            if done:
                break

        if score >= min_score_requiremenent:
            accepted_scores.append(score)
            for data in game_memmory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                else:
                    output = data[1]

                training_data.append([data[0], output])
        
        env.reset()
        scores.append(score)
    training_data = np.array(training_data)
    save_data(training_data, data_dir=data_dir)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print('Training_data lenght:', training_data.shape[0])
    #print(Counter(accepted_scores))

    return training_data


def play_the_game(env, model, goal_steps, data_dir, min_score_requiremenent=100, n_of_games=200, render=True):
    scores = []
    choices = []
    playing_data = []
    
    for each_game in range(n_of_games):
        score = 0
        game_memmory = []
        prev_observation = []
        env.reset()

        for i in range(goal_steps):

            if render and (each_game % 1 == 0):
                env.render()
            
            if len(prev_observation) == 0 :
                action = [np.random.rand() * 4 - 2]
            else:
                action = ((model.predict(prev_observation.reshape(-1, len(prev_observation)))[0]) * 4) - 2

            choices.append(action)
            #print(action)
            new_observation, reward, done, info = env.step(action)
            
            prev_observation = new_observation

            game_memmory.append([new_observation, action])

            score += reward

            if done:
                break

        if score >= min_score_requiremenent:
            for data in game_memmory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                else:
                    output = data[1]

                playing_data.append([data[0], output])

        scores.append(score)


    scores = np.array(scores)
    print('Average score:', mean(scores))
    print('Median score:', median(scores))

    playing_data = np.array(playing_data)
    
    save_data(playing_data, 
              data_dir=data_dir,
              add_data=True)
    return playing_data