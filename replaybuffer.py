import numpy as np


class ReplayBuffer:
    def __init__(self, size, input_shape):
        self.size = size
        self.input_shape = input_shape
        self.s1 = np.zeros([self.size, ] + self.input_shape)
        self.s2 = np.zeros([self.size, ] + self.input_shape)
        self.a = np.zeros([self.size])
        self.r = np.zeros([self.size])
        self.d = np.zeros([self.size])
        self.counter = 0

    def store_transition(self, obs, obs_, action, reward, done):
        memory_idx = self.counter % self.size
        self.s1[memory_idx] = obs
        self.s2[memory_idx] = obs_
        self.a[memory_idx] = action
        self.r[memory_idx] = reward
        self.d[memory_idx] = done
        self.counter += 1

    def sample(self, batch_size):
        rand_idx = np.random.choice(min(self.counter, self.size), batch_size)

        return [self.s1[rand_idx].copy(),
                self.s2[rand_idx].copy(),
                self.a[rand_idx].copy(),
                self.r[rand_idx].copy(),
                self.d[rand_idx].copy()]

    def clear(self):
        self.s1 = np.zeros([self.size, ] + self.input_shape)
        self.s2 = np.zeros([self.size, ] + self.input_shape)
        self.a = np.zeros([self.size])
        self.r = np.zeros([self.size])
        self.d = np.zeros([self.size])
        self.counter = 0

    def get_current_size(self):
        return min(self.size, self.counter)