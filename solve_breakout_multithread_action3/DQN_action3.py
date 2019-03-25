import numpy as np
import tensorflow as tf
from DQN_base import DQN_Base


class DQN(DQN_Base):
    def __init__(self,
                 run_name,
                 input_shape,
                 n_action,
                 gamma,
                 learning_rate,
                 save_path='./result/',
                 record_io=True,
                 record=True,
                 gpu_fraction=None):

        super().__init__(run_name=run_name,
                         input_shape=input_shape,
                         n_action=n_action,
                         gamma=gamma,
                         learning_rate=learning_rate,
                         save_path=save_path,
                         record_io=record_io,
                         record=record,
                         gpu_fraction=gpu_fraction)

    def _build_network(self):
        self.S1 = tf.placeholder(tf.float32, shape=[None]+self.input_shape, name='obs1')
        self.S2 = tf.placeholder(tf.float32, shape=[None]+self.input_shape, name='obs2')
        self.A = tf.placeholder(tf.int32, shape=[None], name='action')
        self.R = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.D = tf.placeholder(tf.float32, shape=[None], name='terminate')

        def network(x, name, trainable):
            initializer = tf.contrib.layers.xavier_initializer()
            c_name = [name, tf.GraphKeys.GLOBAL_VARIABLES]
            weight = {'conv1': tf.get_variable(name+'_w_conv1', [8, 8, 4, 32],
                                               initializer=initializer, collections=c_name, trainable=trainable),

                      'conv2': tf.get_variable(name+'_w_conv2', [4, 4, 32, 64],
                                               initializer=initializer, collections=c_name, trainable=trainable),

                      'conv3': tf.get_variable(name + '_w_conv3', [3, 3, 64, 64],
                                               initializer=initializer, collections=c_name, trainable=trainable),

                      'fc1': tf.get_variable(name+'_w_fc1', [3136, 512],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                      'out': tf.get_variable(name+'_w_out', [512, self.n_action],
                                             initializer=initializer, collections=c_name, trainable=trainable)}

            bias = {'conv1': tf.get_variable(name+'_b_conv1', [32],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                    'conv2': tf.get_variable(name+'_b_conv2', [64],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                    'conv3': tf.get_variable(name + '_b_conv3', [64],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                    'fc1': tf.get_variable(name+'_b_fc1', [512],
                                           initializer=initializer, collections=c_name, trainable=trainable),

                    'out': tf.get_variable(name+'_b_out', [self.n_action],
                                           initializer=initializer, collections=c_name, trainable=trainable)}

            conv1 = tf.nn.relu(tf.nn.conv2d(x, weight['conv1'], strides=[1, 4, 4, 1], padding='VALID') + bias['conv1'])
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight['conv2'], strides=[1, 2, 2, 1], padding='VALID') + bias['conv2'])
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weight['conv3'], strides=[1, 1, 1, 1], padding='VALID') + bias['conv3'])

            flattened = tf.reshape(conv3, shape=[-1, 3136])
            fc1 = tf.nn.relu(tf.matmul(flattened, weight['fc1']) + bias['fc1'])
            out = tf.matmul(fc1, weight['out']) + bias['out']

            Summary = [tf.summary.histogram('w_conv1', weight['conv1']),
                       tf.summary.histogram('w_conv2', weight['conv2']),
                       tf.summary.histogram('w_conv3', weight['conv3']),
                       tf.summary.histogram('w_fc1', weight['fc1']),
                       tf.summary.histogram('w_out', weight['out']),
                       tf.summary.histogram('b_conv1', bias['conv1']),
                       tf.summary.histogram('b_conv2', bias['conv2']),
                       tf.summary.histogram('b_conv3', bias['conv3']),
                       tf.summary.histogram('b_fc1', bias['fc1']),
                       tf.summary.histogram('b_out', bias['out'])]

            return out, Summary

        with tf.name_scope('eval_net'):
            self.Q_eval, Summary1 = network(self.S1, 'eval_net', True)
            self.Q_eval_max_action = tf.argmax(self.Q_eval, axis=1)

        with tf.name_scope('target_net'):
            self.Q_targ, Summary2 = network(self.S2, 'targ_net', False)
            self.Q_targ_max_value = tf.reduce_max(self.Q_targ, axis=1)

        self.Summary_weight = tf.summary.merge(Summary1+Summary2)
        self.A_index = tf.one_hot(self.A, self.n_action)
        self.Q_eval_spec_a = tf.reduce_sum(self.Q_eval * self.A_index, axis=1)
        self.Constant_1 = tf.constant(1, dtype=tf.float32)
        self.Regre_value = self.Q_targ_max_value * self.gamma * (self.Constant_1-self.D) + self.R
        self.Loss = tf.reduce_mean(tf.square(self.Q_eval_spec_a - self.Regre_value))
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)
        # self.Train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)