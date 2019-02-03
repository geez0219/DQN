import numpy as np
import tensorflow as tf
from DQN_base import DQN_base

class DQN(DQN_base):
    def __init__(self,
                 run_name,
                 input_shape,
                 n_action,
                 conv_size,
                 conv1_depth,
                 conv2_depth,
                 fc1_depth,
                 replay_buffer_size=10000,
                 train_epoch=1,
                 train_batch=32,
                 gamma=0.9,
                 epislon_decrease=1/5000,
                 epislon_min=0.025,
                 learning_rate=5e-4
                 ):

        self.conv_size = conv_size
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.fc1_depth = fc1_depth
        super().__init__(run_name=run_name,
                         input_shape=input_shape,
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
        self.S1 = tf.placeholder(tf.float32, shape=[None]+self.input_shape)
        self.S2 = tf.placeholder(tf.float32, shape=[None]+self.input_shape)
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.D = tf.placeholder(tf.float32, shape=[None])


        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

        def maxpool(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


        def network(x, name, trainable):
            initializer = tf.contrib.layers.xavier_initializer()
            c_name = [name, tf.GraphKeys.GLOBAL_VARIABLES]
            weight = {'conv1': tf.get_variable(name+'_w_conv1', [self.conv_size, self.conv_size, self.input_shape[2], self.conv1_depth],
                                               initializer=initializer, collections=c_name, trainable=trainable),

                      'conv2': tf.get_variable(name+'_w_conv2', [self.conv_size, self.conv_size, self.conv1_depth, self.conv2_depth],
                                               initializer=initializer, collections=c_name, trainable=trainable),

                      'fc1': tf.get_variable(name+'_w_fc1', [int((self.input_shape[0]/4)*(self.input_shape[0]/4)*self.conv2_depth), self.fc1_depth],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                      'out': tf.get_variable(name+'_w_out', [self.fc1_depth, self.n_action],
                                             initializer=initializer, collections=c_name, trainable=trainable)}

            bias = {'conv1': tf.get_variable(name+'_b_conv1', [self.conv1_depth],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                    'conv2': tf.get_variable(name+'_b_conv2', [self.conv2_depth],
                                             initializer=initializer, collections=c_name, trainable=trainable),

                    'fc1': tf.get_variable(name+'_b_fc1', [self.fc1_depth],
                                           initializer=initializer, collections=c_name, trainable=trainable),

                    'out': tf.get_variable(name+'_b_out', [self.n_action],
                                           initializer=initializer, collections=c_name, trainable=trainable)}

            conv1 = tf.nn.relu(conv2d(x, weight['conv1']) + bias['conv1'])
            maxpool1 = maxpool(conv1)
            conv2 = tf.nn.relu(conv2d(maxpool1, weight['conv2']) + bias['conv2'])
            maxpool2 = maxpool(conv2)
            flattend = tf.reshape(maxpool2, shape=[-1, int((self.input_shape[0]/4)*(self.input_shape[0]/4)*self.conv2_depth)])
            fc1 = tf.nn.relu(tf.matmul(flattend, weight['fc1']) + bias['fc1'])
            out = tf.matmul(fc1, weight['out']) + bias['out']

            Summary = [tf.summary.histogram('w_conv1', weight['fc1']),
                       tf.summary.histogram('w_conv2', weight['out']),
                       tf.summary.histogram('w_fc1', weight['fc1']),
                       tf.summary.histogram('w_out', weight['out']),
                       tf.summary.histogram('b_fc1', bias['fc1']),
                       tf.summary.histogram('b_out', bias['out']),
                       tf.summary.histogram('b_conv1', bias['fc1']),
                       tf.summary.histogram('b_conv2', bias['out'])]

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
        self.Regre_value = self.Q_targ_max_value * self.gamma * (1-self.D) + self.R
        self.Loss = tf.reduce_mean(tf.square(self.Q_eval_spec_a - self.Regre_value))
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)
