import numpy as np
import gym
import argparse
from solve_breakout.DQN_full import DQN
from environment import Environment
from replaybuffer import ReplayBuffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="the name of the training model", type=str)
    parser.add_argument("-p", "--save_path", help="the save path of checkpoint", type=str, default="./result/")
    parser.add_argument("-t", "--thread_num", help="the number of playing thread", type=int, default=10)
    parser.add_argument("-n", "--game_num", help="the number of training games", type=int, default=500000)
    parser.add_argument("-u", "--update_period", help="the update period of target network", type=int, default=2500)
    parser.add_argument("-e1", "--epsilon_min", help="the epsilon of exploration", type=float, default=2e-3)
    parser.add_argument("-e2", "--epsilon_decrease", help="the epsilon decrease", type=float, default=2e-5)
    parser.add_argument("-g", "--gamma", help="the gamma of DQN learning", type=float, default=0.99)
    parser.add_argument("-l", "--learning_rate", help="the learning rate of DQN training", type=float, default=5e-4)
    parser.add_argument("-d1", "--double_DQN", help="adopt double DQN if call", action='store_true')
    parser.add_argument("-d2", "--dueling_DQN", help="adopt dueling DQN if call", action='store_true')

    arg = parser.parse_args()

    print("---------- argument setting -----------")
    print("run_name: {}".format(arg.run_name))
    print("save_path: {}".format(arg.save_path))
    print("thread_num: {}".format(arg.thread_num))
    print("games_num: {}".format(arg.game_num))
    print("update_period: {}".format(arg.update_period))
    print("epsilon_min: {}".format(arg.epsilon_min))
    print("epsilon_decrease: {}".format(arg.epsilon_decrease))
    print("gamma: {}".format(arg.gamma))
    print("learning_rate: {}".format(arg.learning_rate))
    print("double_DQN: {}".format(arg.double_DQN))
    print("dueling_DQN: {}".format(arg.dueling_DQN))
    print("---------------------------------------")

    record_period = 100
    train_batch = 32
    epsilon = 1
    train_period = 4
    input_shape = [84,84,4]

    env = Environment('BreakoutNoFrameskip-v4', atari_wrapper=True, test=False)
    agent = DQN(run_name=arg.run_name,
                input_shape=input_shape,
                n_action=3,
                gamma=arg.gamma,
                learning_rate=arg.learning_rate,
                save_path=arg.save_path,
                double_DQN=arg.double_DQN,
                dueling_DQN=arg.dueling_DQN,
                record_io=True,
                record=True,
                gpu_fraction=0.3)

    replay_buffer = ReplayBuffer(size=10000, input_shape=input_shape)

    step = 0
    while step < arg.game_num:
        obs = env.reset()
        done = 0
        total_reward = 0
        step = agent.step_move()
        epsilon = max(1 - step * arg.epsilon_decrease, arg.epsilon_min)
        while not done:
            if np.random.uniform(0,1) < epsilon:
                action = agent.random_action()
            else:
                action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action+1)  # because there is only three action
            replay_buffer.store_transition(obs, obs_, action, reward, done)
            total_reward += reward
            obs = obs_

        print('in {}, {}th game: the reward {} '.format(arg.run_name, step, total_reward))

        if step % train_period == 0:
            s1, s2, a, r, d = replay_buffer.sample(batch_size=train_batch)
            if step % record_period == 0:
                loss = agent.train(s1, s2, a, r, d, True)
                agent.log_reward(total_reward)
                agent.save()
            else:
                loss = agent.train(s1, s2, a, r, d, False)
            print('{}th game: the training loss {}'.format(step, loss))

        if step % arg.update_period == 0:
            agent.update_target_network()
