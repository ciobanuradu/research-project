from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import time
import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class dynamic_environment(py_environment.PyEnvironment):

    def __init__(self, actions, gamma, threshold, links, min=0.1, max=0.1, time=1):
        self.links = links
        self.gamma = gamma
        self.threshold = threshold
        self.max = max
        self.delta = (max - min) / time
        self.actions = [[action[0],action[1]] for action in actions]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(actions) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(links + 1,), dtype=np.float64, minimum=[0] * (links + 1), maximum=[links] + [1] * links, name='observation')
        self.state = np.repeat(0, links).astype(np.float64)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.state = np.repeat(0.0, self.links)
        return ts.restart(np.concatenate((np.array(object=[0], dtype=np.float64), np.array(self.state)), dtype=np.float64))

    def _step(self, action):
        # start = time.time()
        if action not in range(len(self.actions)):
            raise ValueError(f'`action` should be in [0, {len(self.actions) - 1}]')

        for i, val in enumerate(self.state):
            if val != 0:
                self.state[i] = (self.state[i] - 1/4) * (np.e ** (-self.gamma)) + 1/4
            if self.state[i] < self.threshold:
                self.state[i] = 0

        if self.gamma < self.max:
            self.gamma += self.delta
        
        self.state.sort()
        for i, val in enumerate(self.state):
            if val == 0:
                self.state[i] = self.actions[action][1] * np.random.binomial(1, self.actions[action][0])
                break
        for i, val in enumerate(self.state):
            if val == 0:
                # end = time.time()
                # print(f"time elapsed: {end - start}")
                return ts.transition(np.concatenate((np.array([len([i for i in self.state if i != 0])]), np.array(self.state)), dtype=np.float64), reward = -1.0, discount = 1.0)
        
        # end = time.time()
        # print(f"time elapsed: {end - start}")
        return ts.termination(np.concatenate((np.array([len([i for i in self.state if i != 0])]), np.array(self.state)), dtype=np.float64), reward=-1.0)


# actions = [[0.9, 0.6],[0.7, 1.0]] #[[p1,F1],[p2,F2]]
# links = 4
# threshold = 0.5
# gamma = 0.1
# print("Creating environment...")
# environment = dynamic_environment(actions, gamma, threshold, links)
# print("Environment created successfully.")
# utils.validate_py_environment(environment, episodes=5)
