import os
import shutil
import numpy as np
import gym
from solve_breakout.DQN import DQN
from environment import Environment

if __name__ == '__main__':
    folder_name = 'Breakout'
    env = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)

    para = [[32,  64, 400]]

    if os.path.exists('./' + folder_name):
        print('the multiple_run folder already exists!')
        print('0: exist ')
        print('1: start over and overwrite')
        print('2: create a new folder')
        mode = int(input('please select the mode:'))

        if mode == 0:
            exit('you select to exist')
        elif mode == 1:
            shutil.rmtree('./{}'.format(folder_name))
            os.makedirs('./{}'.format(folder_name))
        elif mode == 2:
            folder_name = input('please enter a new folder name')
            os.makedirs('./{}'.format(folder_name))
        else:
            raise ValueError('the valid actions are in range [0-2]')

    for j in range(len(para)):
        print('creating model with para:{}'.format(para[j]))

        agent = DQN(run_name='model{}'.format(j),
                    input_shape=[84,84,4],
                    n_action=4,
                    conv_size=5,
                    conv1_depth=para[j][0],
                    conv2_depth=para[j][1],
                    fc1_depth=para[j][2],
                    replay_buffer_size=10000,
                    train_epoch=1,
                    train_batch=32,
                    gamma=1,
                    epislon_decrease=1/50000,
                    epislon_min=0.025,
                    learning_rate=1.5e-4,
                    save_path='./' + folder_name
                    )

        game_play = 100000
        save_period = 10
        train_period = 4
        update_period = 2500
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
        agent.clear_graph()