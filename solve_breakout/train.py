import numpy as np
import gym
from solve_breakout.DQN_copy import DQN
from environment import Environment

if __name__ == '__main__':
    env = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)
    agent = DQN(run_name='solve_breakout_DQN_copy',
                input_shape=[84,84,4],
                n_action=4,
                replay_buffer_size=10000,
                train_epoch=1,
                train_batch=32,
                gamma=0.99,
                learning_rate=5e-4,
                )

    game_play = 1000
    save_period = 100
    train_period = 4
    target_update_period = 2500
    epislon_decrease = 1 / 50000
    epislon_min = 0.025

    for i in range(game_play):
        obs = env.reset()
        done = 0
        total_reward = 0
        step = agent.step_move()
        epislon = max(1 - i * epislon_decrease, epislon_min)
        while not done:
            if np.random.uniform(0,1) < epislon:
                action = agent.random_action()
            else:
                action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, reward, obs_, done)
            total_reward += reward
            obs = obs_

        print('{}th game: the reward {}'.format(step, total_reward))

        if i % train_period == 0:
            if i % save_period == 0:
                loss = agent.train(True)
                agent.log_reward(total_reward)
                agent.save()
            else:
                loss = agent.train(False)
            print('{}th game: the training loss {}'.format(step, loss))

        if i % target_update_period == 0:
            agent.update_target_network()
