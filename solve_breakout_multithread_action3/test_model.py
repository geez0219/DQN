"""
Try to solve_breakout:
1. using copy model
2. reduce the action space to 3
3. Adam optimizer
"""


import numpy as np
import gym
import os
import argparse
import time
from solve_breakout_multithread_action3.DQN_action3 import DQN
from environment import Environment


def play_game(agent, env, epislon, game_play, render):
    reward_avg = 0
    for i in range(game_play):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                time.sleep(1/30)
                env.render()
            if np.random.uniform(0,1) < epislon:
                action = agent.random_action()
            else:
                action = agent.choose_action(obs)
            obs, reward, done,  _ = env.step(action + 1)
            total_reward += reward

        reward_avg += (total_reward - reward_avg)/(i+1) # use moving average to get reward average
        print('in {}th game: the total_reward is: {}, average total_reward is: {}'.format(i, total_reward, reward_avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="the name of the training model", type=str)
    parser.add_argument("-e", "--epislon", help="the epislon of exploration", type=float, default=0.0025)
    parser.add_argument("-n", "--games_num", help='the number of training games', type=int, default=100)
    parser.add_argument("-p", "--load_path", help='the load path of checkpoint', type=str, default='./result/')
    parser.add_argument("-s", "--show", help='whether to show the gameplay screen', action='store_true')

    arg = parser.parse_args()

    print('---------- argument setting -----------')
    print('run_name: {}'.format(arg.run_name))
    print('epislon: {}'.format(arg.epislon))
    print('games_num: {}'.format(arg.games_num))
    print('load_path: {}'.format(arg.load_path))
    print('show: {}'.format(arg.show))
    print('---------------------------------------')

    env = Environment('BreakoutNoFrameskip-v4', test=True, atari_wrapper=True)

    if not os.path.exists('{}{}'.format(arg.load_path, arg.run_name)):
        raise ValueError('{}{} did not exist! '.format(arg.load_path, arg.run_name))

    agent = DQN(run_name=arg.run_name,
                input_shape=[84, 84, 4],
                n_action=3,
                learning_rate=0,
                gamma=0,
                save_path=arg.load_path,
                record_io=False,
                record=False,
                gpu_fraction=0.9)

    agent.load(arg.load_path, arg.run_name)
    play_game(agent, env, arg.epislon, arg.games_num, arg.show)

