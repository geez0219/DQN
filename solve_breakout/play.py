import numpy as np
import gym
import time
from solve_breakout.DQN_copy import DQN
from environment import Environment

if __name__ == '__main__':
    env = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)
    agent = DQN(run_name='test_step',
                input_shape=[84,84,4],
                n_action=4,
                conv_size=5,
                conv1_depth=6,
                conv2_depth=16,
                fc1_depth=400,
                replay_buffer_size=10000,
                train_epoch=1,
                train_batch=32,
                gamma=1,
                epislon_decrease=1/50000,
                epislon_min=0.025,
                learning_rate=5e-4,
                )

    game_play = 1000
    for i in range(game_play):
        obs = env.reset()
        done = 0
        total_reward = 0

        while not done:
            env.render()
            time.sleep(1/30)
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            total_reward += reward
            obs = obs_
        print('{}th game: the reward {}'.format(i, total_reward))

