'''
the code to try OpenAI gym game yourself and help understanding the game.
'''

import gym
import numpy as np
import time
import keyboard
from environment import Environment
import cv2
# parameter setting
game_name = 'BreakoutNoFrameskip-v4'
saved_file_name = 'sar_pair.npy'
esc = 0

#control setting
control = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'esc': 'esc'}
s_list = []
a_list = []
r_list = []

def check_input_range(env):
    obs = env.reset()
    max_, min_ = np.max(obs), np.min(obs)
    print('-------- check input range ------------')
    print('the input shape is: {}'.format(obs.shape))
    print('the input range is: [{} ~ {}]'.format(min_, max_))

def computer_play(save_or_not):
    esc = 0
    while not esc:
        obs = env.reset()
        done = 0
        env.render()
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, _ = env.step(action)
            print('action:{}, reward:{}, done:{}'.format(action, reward, done))
            env.render()
            if save_or_not:
                s_list.append(obs)
                a_list.append(action)
                r_list.append(reward)

            if keyboard.is_pressed('esc'):
                esc = 1

def play_frame(save_or_not):
    while True:
        obs = env.reset()
        done = 0
        env.render()
        while not done:
            action = None
            while action == None:
                for key in control:
                    if keyboard.is_pressed(key):
                        action = control[key]

            if action == 'esc':
                return
            obs_, reward, done, _ = env.step(action)
            print('action:{}, reward:{}, done:{}'.format(action, reward, done))
            # cv2.imshow('0', obs_[:, :, 0])
            # cv2.imshow('1', obs_[:, :, 1])
            # cv2.imshow('2', obs_[:, :, 2])
            # cv2.imshow('3', obs_[:, :, 3])
            # cv2.waitKey(30)


            env.render()
            if save_or_not:
                s_list.append(obs)
                a_list.append(action)
                r_list.append(reward)
            time.sleep(0.1)

def play_realtime(save_or_not):
    while True:
        obs = env.reset()
        done = 0
        env.render()
        while not done:
            action = None
            for key in control:
                if keyboard.is_pressed(key):
                    action = control[key]

            if action == 'esc':
                return
            if action == None:
                action = 0
            obs_, reward, done, _ = env.step(action)
            print('action:{}, reward:{}, done:{}'.format(action, reward, done))
            env.render()
            if save_or_not:
                s_list.append(obs)
                a_list.append(action)
                r_list.append(reward)
            time.sleep(0.1)

if __name__ == '__main__':
    env = Environment(game_name, 0, atari_wrapper=True)

    print('We are playing {}'.format(game_name))
    print('-------- game information --------')
    print('observation space: ', end='')
    print(env.observation_space)
    print('action space: ', end='')
    print(env.action_space)

    try:
        print('action meanings: ', end='')
        print(env.unwrapped.get_action_meanings())
    except:
        print('don\'t know the action meanings')

    check_input_range(env)

    print('--------- game mode select------------')
    print('0: computer randomly choose action')
    print('1: play the game frame by frame')
    print('2: play the game in real time')
    command = int(input('please select the mode:'))
    save_or_not = int(input('Do you want to save the sarsa_pair in file {} (0: no, 1: yes)'.format(saved_file_name)))

    # computer randomly chooses action
    if command == 0:
        computer_play(save_or_not)
    elif command == 1:
        play_frame(save_or_not)
    elif command == 2:
        play_realtime(save_or_not)

    if save_or_not:
        sarsa_pair = [s_list, a_list, r_list]
        with open(saved_file_name, 'wb') as file:
            np.save(file, sarsa_pair)

