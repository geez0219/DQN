import numpy as np
import tensorflow as tf
import os
import shutil


class DQN:
    def __init__(self,
                 run_name,
                 n_feature,
                 n_action,
                 n_l1,
                 replay_buffer_size=10000,
                 train_epoch=1,
                 train_batch=32,
                 gamma=0.9,
                 epislon_decrease=1/5000,
                 epislon_min=0.025,
                 learning_rate=5e-4,
                 ):
        self.run_name = run_name
        self.n_feature = n_feature
        self.n_action = n_action
        self.n_l1 = n_l1
        self.replay_buffer_size = replay_buffer_size
        self.train_epoch = train_epoch
        self.train_batch = train_batch
        self.gamma = gamma
        self.epislon_decrease = epislon_decrease
        self.epislon_min = epislon_min
        self.learning_rate = learning_rate
        self.memory_s, self.memory_a, self.memory_r, self.memory_s2, self.memory_d = [], [], [], [], []

        self._build_network()
        self._build_other()
        self.Sess = tf.Session()
        self.Sess.run(tf.global_variables_initializer())
        if os.path.exists('./' + run_name):
            print('the run directory already exists!')
            print('0: exist ')
            print('1: restored the session from checkPoint ')
            print('2: start over and overwrite')
            print('3: create a new run')
            mode = int(input('please select the mode:'))

            if mode == 0:
                exit('you select to exist')
            elif mode == 1:
                self.load()
            elif mode == 2:
                shutil.rmtree('./{}/{}'.format(self.run_name, 'tensorboard'))
            elif mode == 3:
                self.run_name = input('please enter a new run name')
            else:
                raise ValueError('the valid actions are in range [0-3]')

        self.Writer = tf.summary.FileWriter(self.run_name + '/tensorboard', self.Sess.graph)
        self.update_target_network()

    def _build_network(self):
        self.S1 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.S2 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.D = tf.placeholder(tf.float32, shape=[None])

        def network(input, name, trainable):
            initializer = tf.contrib.layers.xavier_initializer()
            c_name = [name, tf.GraphKeys.GLOBAL_VARIABLES]
            Weight = {'fc1': tf.get_variable(name+'\/w_fc1', [self.n_feature, self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                      'out': tf.get_variable(name+'\/w_out', [self.n_l1, self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

            Bias = {'fc1': tf.get_variable(name+'\/b_fc1', [self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                    'out': tf.get_variable(name+'\/b_out', [self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

            L1 = tf.nn.relu(tf.matmul(input, Weight['fc1']) + Bias['fc1'])
            L2 = tf.matmul(L1, Weight['out']) + Bias['out']

            Summary = [tf.summary.histogram('w_fc1', Weight['fc1']),
                       tf.summary.histogram('w_out', Weight['out']),
                       tf.summary.histogram('b_fc1', Bias['fc1']),
                       tf.summary.histogram('b_out', Bias['out'])]

            return L2, Summary


        with tf.name_scope('eval_net'):
            self.Q_eval, Summary1 = network(self.S1, 'eval_net', True)
            self.Q_eval_max_action = tf.argmax(self.Q_eval, axis=1)

        with tf.name_scope('target_net'):
            self.Q_targ, Summary2 = network(self.S2, 'targ_net', False)
            self.Q_targ_max_value = tf.reduce_max(self.Q_targ, axis=1)

        self.Summary_weight = tf.summary.merge(Summary1+Summary2)
        self.A_index = tf.one_hot(self.A, self.n_action)
        self.Q_eval_spec_a = tf.reduce_sum(self.Q_eval * self.A_index, axis=1)
        self.Regre_value = self.Q_targ_max_value * self.gamma * (1-self.D) + self.R
        self.Loss = tf.reduce_mean(tf.square(self.Q_eval_spec_a - self.Regre_value))
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)

    def _build_other(self):
        self.Saver = tf.train.Saver()

        with tf.name_scope('assign_target_net'):
            t_params = tf.get_collection('targ_net')
            e_params = tf.get_collection('eval_net')
            self.Update_target = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        with tf.name_scope('reflect'):
            self.Loss_reflect = tf.placeholder(tf.float32, shape=None)
            self.Reward_reflect = tf.placeholder(tf.float32, shape=None)

        with tf.name_scope('summary'):
            self.Summary_loss = tf.summary.scalar('loss', self.Loss_reflect)
            self.Summary_reward = tf.summary.scalar('total_reward', self.Reward_reflect)

        with tf.name_scope('step_counter'):
            self.Step = tf.Variable(tf.constant(0), dtype=tf.int32)
            self.Step_move = tf.assign(self.Step, self.Step + tf.constant(1))

    def choose_action(self, obs):
        step = self.Sess.run(self.Step)
        epislon = max(1 - step * self.epislon_decrease, self.epislon_min)

        if np.random.uniform(0,1) < epislon:
            action = np.random.choice(self.n_action)
        else:
            obs = np.expand_dims(obs, axis=0)
            action = self.Sess.run(self.Q_eval_max_action, feed_dict={self.S1: obs})[0]

        return action

    def store_transition(self, obs, action, reward, obs_, done):
        self.memory_s.append(obs)
        self.memory_a.append(action)
        self.memory_r.append(reward)
        self.memory_s2.append(obs_)
        self.memory_d.append(done)
        current_m_size = len(self.memory_s)
        if len(self.memory_s) > self.replay_buffer_size:
            self.memory_s = self.memory_s[current_m_size - self.replay_buffer_size:]
            self.memory_a = self.memory_a[current_m_size - self.replay_buffer_size:]
            self.memory_r = self.memory_r[current_m_size - self.replay_buffer_size:]
            self.memory_s2 = self.memory_s2[current_m_size - self.replay_buffer_size:]
            self.memory_d = self.memory_d[current_m_size - self.replay_buffer_size:]

    def train(self):
        current_m_size = len(self.memory_s)
        total_loss = np.zeros(self.train_epoch)

        for i in range(self.train_epoch):
            rand_idx = np.random.permutation(current_m_size)
            s1_array = np.array(self.memory_s)[rand_idx]
            a_array = np.array(self.memory_a)[rand_idx]
            r_array = np.array(self.memory_r)[rand_idx]
            s2_array = np.array(self.memory_s2)[rand_idx]
            d_array = np.array(self.memory_d)[rand_idx]

            for j in range(0, current_m_size-self.train_batch, self.train_batch):
                _, loss = self.Sess.run([self.Train, self.Loss], feed_dict={self.S1: s1_array[j:j+self.train_batch],
                                                                            self.A: a_array[j:j+self.train_batch],
                                                                            self.R: r_array[j:j+self.train_batch],
                                                                            self.S2: s2_array[j:j+self.train_batch],
                                                                            self.D: d_array[j:j+self.train_batch]})
                total_loss[i] += loss
        total_loss_mean = np.mean(total_loss)

        result1, result2, step = self.Sess.run([self.Summary_loss, self.Summary_weight,self.Step], feed_dict={self.Loss_reflect: total_loss_mean})
        self.Writer.add_summary(result1, step)
        self.Writer.add_summary(result2, step)

        return total_loss_mean

    def save(self):
        self.Saver.save(self.Sess, './{}/{}.ckpt'.format(self.run_name, self.run_name))

    def load(self):
        self.Saver.restore(self.Sess, './{}/{}.ckpt'.format(self.run_name, self.run_name))

    def clear_replay_buffer(self):
        self.memory_s, self.memory_a, self.memory_r, self.memory_s2, self.memory_d = [], [], [], [], []

    def step_move(self):
        step = self.Sess.run(self.Step_move)
        return step

    def log_reward(self, reward):
        result, step = self.Sess.run([self.Summary_reward, self.Step], feed_dict={self.Reward_reflect: reward})
        self.Writer.add_summary(result, step)

    def update_target_network(self):
        self.Sess.run(self.Update_target)
