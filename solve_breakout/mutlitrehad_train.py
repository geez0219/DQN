import numpy as np
import gym
import threading
from solve_breakout.DQN_multithread import DQN
from environment import Environment
from replay_buffer import replay_buffer

env = Environment('BreakoutNoFrameskip-v4', 0, atari_wrapper=True)
game_play = 200000
# save_period = 1000
run_name = 'solve_breakout_DQN_copy_multithread'
train_period = 4
target_update_period = 2500
record_period = 100
replay_buffer_size = 10000
epislon_decrease = 1 / 50000
epislon_min = 0.025
epislon = 1
train_batch = 1024

replay_buffer = replay_buffer(size=replay_buffer_size, input_shape=[84, 84, 4])


agent_train = DQN(run_name=run_name,
                  input_shape=[84, 84, 4],
                  n_action=4,
                  train_epoch=1,
                  train_batch=train_batch,
                  gamma=0.99,
                  learning_rate=5e-4,
                  record_io=True,
                  save_path='./result/',
                  gpu_fraction=0.45)

agent_play = DQN(run_name=run_name,
                 input_shape=[84, 84, 4],
                 n_action=4,
                 train_epoch=1,
                 train_batch=train_batch,
                 gamma=0.99,
                 learning_rate=5e-4,
                 record_io=False,
                 save_path='./result/',
                 gpu_fraction=0.45)

lock = threading.Lock()

class TrainThread(threading.Thread):
    def run(self):
        for i in range(target_update_period):
            step = agent_train.step_move()
            lock.acquire()
            s1, s2, a, r, d = replay_buffer.sample(train_batch)
            lock.release()
            if i % record_period==0:
                agent_train.train(s1, s2, a, r, d, True)
            else:
                agent_train.train(s1, s2, a, r, d, False)
            # print('train_thread:{}'.format(i))
        agent_train.update_target_network()
        agent_train.save()

class PlayThread(threading.Thread):
    def run(self):
        global epislon
        for i in range(target_update_period):
            step = agent_play.step_move()
            obs = env.reset()
            done = 0
            game_reward = 0
            while not done:
                if np.random.uniform(0,1) < epislon:
                    action = agent_play.random_action()
                else:
                    action = agent_play.choose_action(obs)

                obs_, reward, done, _ = env.step(action)

                lock.acquire()
                replay_buffer.store_transition(obs, obs_, action, reward, done)
                lock.release()
                obs = obs_
                game_reward += reward
                epislon = max(epislon-epislon_decrease, epislon_min)

            if i % record_period == 0:
                agent_play.log_reward(game_reward)

            # print('play_thread:{}'.format(i))


if __name__ == '__main__':
    for i in range(3):
        step = agent_play.step_move()
        obs = env.reset()
        done = 0
        while not done:
            action = agent_play.random_action()
            obs_, reward, done, _ = env.step(action)
            replay_buffer.store_transition(obs, obs_, action, reward, done)
            obs = obs_

    for i in range(int(game_play/target_update_period)):
        train_thread = TrainThread()
        play_thread = PlayThread()
        train_thread.start()
        play_thread.start()
        train_thread.join()
        play_thread.join()
        print('finish training and playing thread {}'.format(i))
        agent_play.load(load_path='./result/', run_name=run_name)
