"""
the network is based on paper "Human-level control through deep reinforcement learning"
1. have function of Double DQN and Dueling DQN

"""

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
                 double_DQN=False,
                 dueling_DQN=False,
                 record_io=True,
                 record=True,
                 gpu_fraction=None):

        self.double_DQN = double_DQN
        self.dueling_DQN = dueling_DQN
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

            conv1 = tf.layers.conv2d(inputs=x,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     strides=[4, 4],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer,
                                     trainable=trainable,
                                     name=name + '_conv1')

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=64,
                                     kernel_size=[4, 4],
                                     strides=[2, 2],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer,
                                     trainable=trainable,
                                     name=name + '_conv2')

            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=64,
                                     kernel_size=[3, 3],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer,
                                     trainable=trainable,
                                     name=name + '_conv3')

            flattened = tf.layers.flatten(conv3)

            if self.dueling_DQN:
                fc1_a = tf.layers.dense(inputs=flattened,
                                        units=512,
                                        activation=tf.nn.relu,
                                        kernel_initializer=initializer,
                                        trainable=trainable,
                                        name=name + '_fc1_a')

                fc1_s = tf.layers.dense(inputs=flattened,
                                        units=512,
                                        activation=tf.nn.relu,
                                        kernel_initializer=initializer,
                                        trainable=trainable,
                                        name=name + '_fc1_s')

                out_a = tf.layers.dense(inputs=fc1_a,
                                        units=self.n_action,
                                        activation=None,
                                        kernel_initializer=initializer,
                                        trainable=trainable,
                                        name=name + '_out_a')

                out_s = tf.layers.dense(inputs=fc1_s,
                                        units=1,
                                        activation=None,
                                        kernel_initializer=initializer,
                                        trainable=trainable,
                                        name=name + '_out_s')

                out = out_a - tf.reduce_mean(out_a, axis=1, keepdims=True) + out_s

            else:  # not dueling DQN
                fc1 = tf.layers.dense(inputs=flattened,
                                      units=512,
                                      activation=tf.nn.relu,
                                      kernel_initializer=initializer,
                                      trainable=trainable,
                                      name=name + '_fc1')

                out = tf.layers.dense(inputs=fc1,
                                      units=self.n_action,
                                      activation=None,
                                      kernel_initializer=initializer,
                                      trainable=trainable,
                                      name=name + '_out')
            return out

        if self.double_DQN:
            self.S_concat = tf.concat([self.S1, self.S2], axis=0)
            with tf.name_scope('eval_net'):
                self.S_concat_eval = network(self.S_concat, 'eval_net', True)
                self.Q_eval_S1, self.Q_eval_S2 = tf.split(self.S_concat_eval, 2, axis=0)
                self.Q_eval_S1_max_action = tf.argmax(self.Q_eval_S1, axis=1, name="Q_eval_S1_max_action")
                self.Q_eval_S2_max_action = tf.argmax(self.Q_eval_S2, axis=1, name="Q_eval_S2_max_action")
                self.Q_eval_S2_max_onehot = tf.one_hot(self.Q_eval_S2_max_action, self.n_action, axis=1)

            with tf.name_scope('target_net'):
                self.Q_targ_S2 = network(self.S2, 'targ_net', False)
                self.Q_targ_value = tf.stop_gradient(tf.reduce_sum(self.Q_targ_S2 * self.Q_eval_S2_max_onehot, axis=1))

        else:
            with tf.name_scope('eval_net'):
                self.Q_eval_S1 = network(self.S1, 'eval_net', True)
                self.Q_eval_S1_max_action = tf.argmax(self.Q_eval_S1, axis=1)

            with tf.name_scope('target_net'):
                self.Q_targ_S2 = network(self.S2, 'targ_net', False)
                self.Q_targ_value = tf.reduce_max(self.Q_targ_S2, axis=1)

        self.A_index = tf.one_hot(self.A, self.n_action)
        self.Q_eval_spec_a = tf.reduce_sum(self.Q_eval_S1 * self.A_index, axis=1)
        self.Constant_1 = tf.constant(1, dtype=tf.float32)
        self.Regre_value = self.Q_targ_value * self.gamma * (self.Constant_1-self.D) + self.R
        self.Loss = tf.reduce_mean(tf.square(self.Q_eval_spec_a - self.Regre_value))
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)
