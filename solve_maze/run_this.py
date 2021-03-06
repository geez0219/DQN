import numpy as np

from solve_maze.DQN import DQN
from solve_maze.maze import Maze

def print_all_Q_value(agent):
    stateList = [[x,y] for y in np.arange(-0.5, 0.5, 0.25) for x in np.arange(-0.5, 0.5, 0.25)]
    stateList = np.array(stateList, dtype=np.float32)

    qValue = agent.Sess.run(agent.Q_eval, feed_dict={agent.S1: stateList})

    for i in range(stateList.shape[0]):
        print('the state:{} has Qvalue:{}'.format(stateList[i], qValue[i]))

if __name__ == '__main__':
    env = Maze()
    agent = DQN(run_name='Maze',
                input_shape=[2],
                n_action=4,
                n_l1=10,
                replay_buffer_size=10000,
                train_epoch=1,
                train_batch=32,
                gamma=0.7,
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
            obs_, reward, done = env.step(action)
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
            print_all_Q_value(agent)
