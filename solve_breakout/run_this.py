import numpy as np
import gym
from solve_breakout.DQN import DQN
from environment import Environment

if __name__ == '__main__':
    env = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)
    agent = DQN(run_name='Breakout',
                input_shape=[84,84,4],
                n_action=4,
                conv_size=5,
                conv1_depth=6,
                conv2_depth=16,
                fc1_depth=400,
                replay_buffer_size=10000,
                train_epoch=1,
                train_batch=32,
                gamma=0.9,
                epislon_decrease=1/5000,
                epislon_min=0.025,
                learning_rate=5e-4,
                )

    game_play = 10000
    save_period = 10
    train_period = 4
    update_period = 1000
    for i in range(game_play):
        obs = env.reset()
        done = 0
        total_reward = 0
        step = agent.step_move()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, reward, obs_, done)
            total_reward += reward
            obs = obs_

        print('{}th game: the reward {}'.format(step, total_reward))

        if i % train_period == 0:
            loss = agent.train()
            agent.log_reward(total_reward)
            print('{}th game: the training loss {}'.format(step, loss))

        if i % save_period == 0:
            agent.save()

        if i % update_period == 0:
            agent.update_target_network()
