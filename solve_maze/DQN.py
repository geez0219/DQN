import numpy as np
import tensorflow as tf
import os
import shutil
from DQN_base import DQN_base

class DQN(DQN_base):
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
                 learning_rate=5e-4
                 ):

        self.n_l1 = n_l1
        super().__init__(run_name=run_name,
                         n_feature=n_feature,
                         n_action=n_action,
                         replay_buffer_size=replay_buffer_size,
                         train_epoch=train_epoch,
                         train_batch=train_batch,
                         gamma=gamma,
                         epislon_decrease=epislon_decrease,
                         epislon_min=epislon_min,
                         learning_rate=learning_rate
                         )


    def _build_network(self):
        self.S1 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.S2 = tf.placeholder(tf.float32, shape=[None, self.n_feature])
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.D = tf.placeholder(tf.float32, shape=[None])

        def network(input, name, trainable):
            initializer = tf.contrib.layers.xavier_initializer()
            c_name = [name, tf.GraphKeys.GLOBAL_VARIABLES]
            Weight = {'fc1': tf.get_variable(name+'_w_fc1', [self.n_feature, self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                      'out': tf.get_variable(name+'_w_out', [self.n_l1, self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

            Bias = {'fc1': tf.get_variable(name+'_b_fc1', [self.n_l1], initializer=initializer, collections=c_name, trainable=trainable),
                    'out': tf.get_variable(name+'_b_out', [self.n_action], initializer=initializer, collections=c_name, trainable=trainable)}

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

