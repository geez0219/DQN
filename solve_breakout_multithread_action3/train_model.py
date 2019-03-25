"""
Try to solve_breakout:
1. using copy model
2. reduce the action space to 3
3. Adam optimizer
"""

import numpy as np
import gym
import os
import shutil
import threading
import time
import argparse
from solve_breakout_multithread_action3.DQN_action3 import DQN
from environment import Environment
from replaybuffer import ReplayBuffer

record_period = 100
train_batch = 32
epislon = 1
lock = threading.Lock()
replay_buffer = ReplayBuffer(size=100000, input_shape=[84, 84, 4])


def run_train_thread(agent, arg):
    for i in range(arg.update_period):
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
        # print("train_thread:{}".format(i))
    agent.update_target_network()
    agent.save()
    # agent.model_pos_check()


def run_play_thread(agent, env, epislon, thread_idx, record, arg):
    if record is True:
        # play with absolute greedy algorithm and log the result
        for i in range(int(arg.update_period / arg.thread_num)):
            for j in range(arg.thread_num):
                agent.step_move()
            obs = env.reset()
            done = 0
            game_reward = 0
            while not done:
                if np.random.uniform(0, 1) < epislon:
                    action = agent.random_action()
                else:
                    action = agent.choose_action(obs)
                obs_, reward, done, _ = env.step(action+1)  # for action 3
                lock.acquire()
                replay_buffer.store_transition(obs, obs_, action, reward, done)
                lock.release()
                obs = obs_
                game_reward += reward

            if i % record_period == 0:
                agent.log_reward(game_reward)
            # print("play_thread{}:{}".format(thread_idx, i))

    else:
        # play with epislon greedy algorithm and doesn't log the result
        for i in range(int(arg.update_period / arg.thread_num)):
            obs = env.reset()
            done = 0
            while not done:
                if np.random.uniform(0, 1) < epislon:
                    action = agent.random_action()
                else:
                    action = agent.choose_action(obs)

                obs_, reward, done, _ = env.step(action+1) # for action 3
                lock.acquire()
                replay_buffer.store_transition(obs, obs_, action, reward, done)
                lock.release()
                obs = obs_
            # print("play_thread{}:{}".format(thread_idx, i))


def check_run_file(arg):
    if os.path.exists("{}{}".format(arg.save_path, arg.run_name)):
        print("the run directory [{}{}] already exists!".format(arg.save_path, arg.run_name))
        print("0: exist ")
        print("1: restored the session from checkPoint ")
        print("2: start over and overwrite")
        print("3: create a new run")
        mode = int(input("please select the mode:"))

        if mode == 0:
            exit("you select to exist")

        elif mode == 2:
            shutil.rmtree("{}{}".format(arg.save_path, arg.run_name))
            os.makedirs("{}{}".format(arg.save_path, arg.run_name))
            shutil.copyfile(__file__, "{}{}/copy_code.py".format(arg.save_path, arg.run_name))

        elif mode == 3:
            arg.run_name = input("please enter a new run name:")
            return check_run_file(arg)

        elif mode > 3 or mode < 0:
            raise ValueError("the valid actions are in range [0-3]")
    else:
        print("create run directory [{}{}]".format(arg.save_path, arg.run_name))
        mode = 2
        os.makedirs("{}{}".format(arg.save_path, arg.run_name))
        shutil.copyfile(__file__, "{}{}/copy_code.py".format(arg.save_path, arg.run_name))

    return mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="the name of the training model", type=str)
    parser.add_argument("-p", "--save_path", help="the save path of checkpoint", type=str, default="./result/")
    parser.add_argument("-t", "--thread_num", help="the number of playing thread", type=int, default=10)
    parser.add_argument("-n", "--game_num", help="the number of training games", type=int, default=500000)
    parser.add_argument("-u", "--update_period", help="the update period of target network", type=int, default=2500)
    parser.add_argument("-e1", "--epislon_min", help="the epislon of exploration", type=float, default=2e-3)
    parser.add_argument("-e2", "--epislon_decrease", help="the epislon decrease", type=float, default=2e-5)
    parser.add_argument("-g", "--gamma", help="the gamma of DQN learning", type=float, default=0.99)
    parser.add_argument("-l", "--learning_rate", help="the learning rate of DQN training", type=float, default=5e-4)

    arg = parser.parse_args()

    print("---------- argument setting -----------")
    print("run_name: {}".format(arg.run_name))
    print("save_path: {}".format(arg.save_path))
    print("thread_num: {}".format(arg.thread_num))
    print("games_num: {}".format(arg.game_num))
    print("update_period: {}".format(arg.update_period))
    print("epislon_min: {}".format(arg.epislon_min))
    print("epislon_decrease: {}".format(arg.epislon_decrease))
    print("gamma: {}".format(arg.gamma))
    print("learning_rate: {}".format(arg.learning_rate))
    print("---------------------------------------")

    # record file io
    mode = check_run_file(arg)

    #declare all agent and environment
    env = [0 for i in range(arg.thread_num)]
    agent_play = [0 for i in range(arg.thread_num)]
    agent_train = DQN(run_name=arg.run_name,
                      input_shape=[84, 84, 4],
                      n_action=3,
                      gamma=arg.gamma,
                      learning_rate=arg.learning_rate,
                      record_io=False,
                      record=True,
                      save_path=arg.save_path,
                      gpu_fraction=0.45)
    if mode == 1:
        agent_train.load(arg.save_path, arg.run_name)

    for i in range(arg.thread_num):
        env[i] = Environment("BreakoutNoFrameskip-v4", test=False, atari_wrapper=True)
        agent_play[i] = DQN(run_name=arg.run_name,
                            input_shape=[84, 84, 4],
                            n_action=3,
                            gamma=0,
                            learning_rate=0,
                            record_io=False,
                            record=True if i == 0 else False,
                            save_path=arg.save_path,
                            gpu_fraction=0.45 / arg.thread_num)
        if mode == 1:
            agent_play[i].load(arg.save_path, arg.run_name)

    # start training and playing loop
    play_thread = [0 for i in range(arg.thread_num)]
    for i in range(int(arg.game_num/arg.thread_num)):
        start_time1 = time.time()
        for j in range(arg.thread_num):
            play_thread[j] = threading.Thread(target=run_play_thread,
                                              args=(agent_play[j], env[j], epislon, j, True if j == 0 else False, arg),
                                              name="play_thread{}".format(j))
            play_thread[j].start()

        start_time2 = time.time()
        run_train_thread(agent_train, arg)

        for j in range(arg.thread_num):
            play_thread[j].join()

        start_time3 = time.time()
        for j in range(arg.thread_num):
            agent_play[j].load(load_path=arg.save_path, run_name=arg.run_name)

        print("finish training and playing thread {}. thread creating time: {}, thread running time: {} agent loading time: {}"
              .format(i, start_time2 - start_time1, start_time3 - start_time2, time.time() - start_time3))

        epislon = max(epislon - arg.epislon_decrease * arg.update_period, arg.epislon_min)