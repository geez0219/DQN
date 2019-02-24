import numpy as np
import gym
import os
import shutil
import threading
from solve_breakout.DQN_multithread import DQN
from environment import Environment
from replaybuffer import ReplayBuffer

game_play = 500000
run_name = 'solve_breakout_DQN_copy_multithread6'
save_path = './result/'
target_update_period = 2500
record_period = 100
replay_buffer_size = 100000
epislon_decrease = 1 / 50000
epislon_min = 0.025
epislon = 1
train_batch = 32
play_thread_num = 10
learning_rate = 5e-4

lock = threading.Lock()
replay_buffer = ReplayBuffer(size=replay_buffer_size, input_shape=[84, 84, 4])


def run_train_thread(agent):
    for i in range(target_update_period):
        step = agent.step_move()
        lock.acquire()
        if replay_buffer.get_current_size() > 0:
            s1, s2, a, r, d = replay_buffer.sample(train_batch)
            lock.release()
            if i % record_period == 0:
                agent.train(s1, s2, a, r, d, True)
            else:
                agent.train(s1, s2, a, r, d, False)

        else:
            lock.release()
        # print('train_thread:{}'.format(i))
    agent.update_target_network()
    agent.save()


def run_play_thread(agent, env, epislon, thread_idx, record):
    if record is True:
        for i in range(int(target_update_period / play_thread_num)):
            for j in range(play_thread_num):
                agent.step_move()
            obs = env.reset()
            done = 0
            game_reward = 0
            while not done:
                if np.random.uniform(0, 1) < epislon:
                    action = agent.random_action()
                else:
                    action = agent.choose_action(obs)

                obs_, reward, done, _ = env.step(action)

                lock.acquire()
                replay_buffer.store_transition(obs, obs_, action, reward, done)
                lock.release()
                obs = obs_
                game_reward += reward

            if i % record_period == 0:
                agent.log_reward(game_reward)
    else:
        for i in range(int(target_update_period / play_thread_num)):
            obs = env.reset()
            done = 0
            while not done:
                if np.random.uniform(0, 1) < epislon:
                    action = agent.random_action()
                else:
                    action = agent.choose_action(obs)

                obs_, reward, done, _ = env.step(action)
                lock.acquire()
                replay_buffer.store_transition(obs, obs_, action, reward, done)
                lock.release()
                obs = obs_

if __name__ == '__main__':
    # record file io
    if os.path.exists('{}{}'.format(save_path, run_name)):
        print('the run directory already exists!')
        print('0: exist ')
        print('1: restored the session from checkPoint ')
        print('2: start over and overwrite')
        print('3: create a new run')
        mode = int(input('please select the mode:'))

        if mode == 0:
            exit('you select to exist')
        elif mode == 2:
            shutil.rmtree('{}{}'.format(save_path, run_name))
        elif mode == 3:
            run_name = input('please enter a new run name')
        elif mode > 3 or mode < 0:
            raise ValueError('the valid actions are in range [0-3]')
    else:
        mode = 2

    # declare all agent and environment
    env = [0] * play_thread_num
    agent_play = [0] * play_thread_num
    agent_train = DQN(run_name=run_name,
                      input_shape=[84, 84, 4],
                      n_action=4,
                      train_epoch=1,
                      train_batch=train_batch,
                      gamma=0.99,
                      learning_rate=learning_rate,
                      record_io=False,
                      record=True,
                      save_path=save_path,
                      gpu_fraction=0.45)
    if mode == 1:
        agent_train.load(save_path, run_name)

    for i in range(play_thread_num):
        env[i] = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)
        agent_play[i] = DQN(run_name=run_name,
                            input_shape=[84, 84, 4],
                            n_action=4,
                            train_epoch=1,
                            train_batch=train_batch,
                            gamma=0.99,
                            learning_rate=learning_rate,
                            record_io=False,
                            record=True if i == 0 else False,
                            save_path=save_path,
                            gpu_fraction=0.45 / play_thread_num)
        if mode == 1:
            agent_play[i].load(save_path, run_name)

    # start training and playing loop
    play_thread = [0] * play_thread_num
    for i in range(int(game_play/target_update_period)):

        for j in range(play_thread_num):
            play_thread[j] = threading.Thread(target=run_play_thread,
                                              args=(agent_play[j], env[j], epislon, j, True if j == 0 else False),
                                              name='play_thread{}'.format(j))
            play_thread[j].start()

        run_train_thread(agent_train)

        for j in range(play_thread_num):
            play_thread[j].join()

        print('finish training and playing thread {}'.format(i))
        epislon = max(epislon - epislon_decrease*target_update_period, epislon_min)

        for j in range(play_thread_num):
            agent_play[j].load(load_path=save_path, run_name=run_name)
