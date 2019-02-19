import numpy as np
import tensorflow as tf
import os
import shutil

class multithread_base:
    def __init__(self,
                 run_name,
                 input_shape,
                 n_action,
                 train_epoch,
                 train_batch,
                 gamma,
                 learning_rate,
                 save_path,
                 record_io,
                 gpu_fraction
                 ):

        self.run_name = run_name
        self.input_shape = input_shape
        self.n_action = n_action
        self.train_epoch = train_epoch
        self.train_batch = train_batch
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.record_io = record_io

        if save_path[-1] != '/':
            self.save_path = save_path + '/'
        else:
            self.save_path = save_path
        
        if self.record_io is True:
            self.deal_record_file()
        with tf.Graph().as_default():
            self._build_network()
            self._build_other()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            self.Sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.Sess.run(tf.global_variables_initializer())

        self.Writer = tf.summary.FileWriter(self.save_path + self.run_name + '/tensorboard', self.Sess.graph)
        self.update_target_network()

    def deal_record_file(self):
        if os.path.exists('{}{}'.format(self.save_path, self.run_name)):
            print('the run directory already exists!')
            print('0: exist ')
            print('1: restored the session from checkPoint ')
            print('2: start over and overwrite')
            print('3: create a new run')
            mode = int(input('please select the mode:'))

            if mode == 0:
                exit('you select to exist')
            elif mode == 1:
                self.load(self.save_path, self.run_name)
            elif mode == 2:
                shutil.rmtree('{}{}'.format(self.save_path, self.run_name))
            elif mode == 3:
                self.run_name = input('please enter a new run name')
            else:
                raise ValueError('the valid actions are in range [0-3]')

    def _build_network(self):
        '''
        the template of the _build_network
        ----------------------------------------------------------------------------------------------------------
        self.S1 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.S2 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.D = tf.placeholder(tf.float32, shape=[None])


        def network(input, name, trainable):
            initializer = tf.contrib.layers.xavier_initializer()
            c_name = [name, tf.GraphKeys.GLOBAL_VARIABLES]

            --------------------------------- change start here ---------------------------------------------------
            Weight = {'fc1': tf.get_variable(name+'_w_fc1', [self.n_feature, self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                      'out': tf.get_variable(name+'_w_out', [self.n_l1, self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

            Bias = {'fc1': tf.get_variable(name+'_b_fc1', [self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                    'out': tf.get_variable(name+'_b_out', [self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

            L1 = tf.nn.relu(tf.matmul(input, Weight['fc1']) + Bias['fc1'])
            output = tf.matmul(L1, Weight['out']) + Bias['out']

            Summary = [tf.summary.histogram('w_fc1', Weight['fc1']),
                       tf.summary.histogram('w_out', Weight['out']),
                       tf.summary.histogram('b_fc1', Bias['fc1']),
                       tf.summary.histogram('b_out', Bias['out'])]

            --------------------------------- change stop here ---------------------------------------------------
            return output, Summary


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
        '''

        raise NotImplementedError("Subclass should implement this")

    def _build_other(self):
        with tf.name_scope('assign_target_net'):
            t_params = tf.get_collection('targ_net')
            e_params = tf.get_collection('eval_net')
            self.Update_target = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        with tf.name_scope('reflect'):
            self.Loss_reflect = tf.placeholder(tf.float32, shape=None)
            self.Reward_reflect = tf.placeholder(tf.float32, shape=None)

        with tf.name_scope('step_counter'):
            self.Step = tf.Variable(tf.constant(0), dtype=tf.int32)
            self.Step_move = tf.assign(self.Step, self.Step + tf.constant(1))

        with tf.name_scope('summary'):
            self.Summary_loss = tf.summary.scalar('loss', self.Loss_reflect)
            self.Summary_reward = tf.summary.scalar('total_reward', self.Reward_reflect)

        self.Saver = tf.train.Saver()

    def choose_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        action = self.Sess.run(self.Q_eval_max_action, feed_dict={self.S1: obs})[0]

        return action

    def random_action(self):
        return np.random.choice(self.n_action)

    def train(self, s1, s2, a, r, d, record):
        _, loss = self.Sess.run([self.Train, self.Loss], feed_dict={self.S1: s1,
                                                                    self.A: a,
                                                                    self.R: r,
                                                                    self.S2: s2,
                                                                    self.D: d})

        if record is True:
            result1, result2, step = self.Sess.run([self.Summary_loss, self.Summary_weight, self.Step], feed_dict={self.Loss_reflect: loss})
            self.Writer.add_summary(result1, step)
            self.Writer.add_summary(result2, step)

        return loss

    def save(self):
        self.Saver.save(self.Sess, '{}{}/{}.ckpt'.format(self.save_path, self.run_name, self.run_name))

    def load(self, load_path=None, run_name=None):
        if load_path is None:
            load_path = self.save_path
        if run_name is None:
            run_name = self.run_name
        self.Saver.restore(self.Sess, '{}{}/{}.ckpt'.format(load_path, run_name, run_name))

    def step_move(self):
        step = self.Sess.run(self.Step_move)
        return step

    def log_reward(self, reward):
        result, step = self.Sess.run([self.Summary_reward, self.Step], feed_dict={self.Reward_reflect: reward})
        self.Writer.add_summary(result, step)

    def update_target_network(self):
        self.Sess.run(self.Update_target)

    def close_model(self):
        self.Sess.close()

