
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from gym import spaces


class MockOpenAIGymDiscrete(object):
    _MOCK_STATE = np.ones((3, ), dtype=float)
    _MOCK_STATE_TYPE = "continuous"
    _MOCK_REWARD = 0.
    _MOCK_DONE = False

    _MOCK_ACTION_SIZE = 2
    _MOCK_ACTION_TYPE = "discrete"
    _MOCK_ACTION_DESCRIPTIONS = None

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=self._MOCK_STATE.shape)

        self.action_space = spaces.Discrete(n=self._MOCK_ACTION_SIZE)
        self._MOCK_ACTION_DESCRIPTIONS = repr(self.action_space)

    def reset(self, train_mode=True):
        self._train_mode = train_mode
        return self._MOCK_STATE

    def step(self, action):
        self._action = action
        return self._MOCK_STATE, self._MOCK_REWARD, self._MOCK_DONE, "INFO"

    def render(self):
        pass


class MockOpenAIGymContinuous(object):
    _MOCK_STATE = np.ones((3, ), dtype=float)
    _MOCK_STATE_TYPE = "continuous"
    _MOCK_REWARD = 0.
    _MOCK_DONE = False

    _MOCK_ACTION_SIZE = (4, 2)
    _MOCK_ACTION_TYPE = "continuous"
    _MOCK_ACTION_DESCRIPTIONS = None

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=self._MOCK_STATE.shape)

        self.action_space = spaces.Box(
            low=-10, high=10, shape=self._MOCK_ACTION_SIZE)
        self._MOCK_ACTION_DESCRIPTIONS = repr(self.action_space)

    def reset(self, train_mode=True):
        self._train_mode = train_mode
        return self._MOCK_STATE

    def step(self, action):
        self._action = action
        return self._MOCK_STATE, self._MOCK_REWARD, self._MOCK_DONE, "INFO"

    def render(self):
        pass
