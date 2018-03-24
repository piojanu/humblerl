from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

class MockUnityEnvironmentVector(object):
    _MOCK_STATE = np.ones((3, ), dtype=float)
    _MOCK_STATE_TYPE = "continuous"
    _MOCK_REWARD = 0.
    _MOCK_DONE = False

    _MOCK_ACTION_SIZE = 2
    _MOCK_ACTION_TYPE = "discrete"
    _MOCK_ACTION_DESCRIPTIONS = ["MOCK_UP", "MOCK_DOWN"]

    _BRAIN_NAMES = ("DEFAULT_BRAIN", )

    class _BrainInfo:

        def __init__(self):
            # List of numpy arrays (num_cameras x height x with x channels)
            self.vector_observations = (MockUnityEnvironmentVector._MOCK_STATE, )
            self.rewards = (MockUnityEnvironmentVector._MOCK_REWARD, )
            self.local_done = (MockUnityEnvironmentVector._MOCK_DONE, )

    class _BrainParameters:

        def __init__(self):
            self.vector_action_space_size = MockUnityEnvironmentVector._MOCK_ACTION_SIZE
            self.vector_action_space_type = MockUnityEnvironmentVector._MOCK_ACTION_TYPE
            self.vector_action_descriptions = MockUnityEnvironmentVector._MOCK_ACTION_DESCRIPTIONS

            self.vector_observation_space_size = MockUnityEnvironmentVector._MOCK_STATE.shape
            self.vector_observation_space_type = MockUnityEnvironmentVector._MOCK_STATE_TYPE

    def __init__(self):
        self.brain_names = self._BRAIN_NAMES
        self.brains = {self.brain_names[0]: self._BrainParameters(), }

    def reset(self, train_mode=True):
        self._train_mode = train_mode
        return {self.brain_names[0]: self._BrainInfo(), }

    def step(self, action):
        self._action = action
        return {self.brain_names[0]: self._BrainInfo(), }

class MockUnityEnvironmentVisual(object):
    _MOCK_STATE = np.ones((5, 4, 3), dtype=float)
    _MOCK_STATE_TYPE = "observation"
    _MOCK_REWARD = 0.
    _MOCK_DONE = False

    _MOCK_ACTION_SIZE = 2
    _MOCK_ACTION_TYPE = "discrete"
    _MOCK_ACTION_DESCRIPTIONS = ["MOCK_UP", "MOCK_DOWN"]

    _BRAIN_NAMES = ("DEFAULT_BRAIN", )

    class _BrainInfo:

        def __init__(self):
            # List of numpy arrays (num_cameras x height x with x channels)
            self.visual_observations = (np.expand_dims(MockUnityEnvironmentVisual._MOCK_STATE, axis=0), )
            self.rewards = (MockUnityEnvironmentVisual._MOCK_REWARD, )
            self.local_done = (MockUnityEnvironmentVisual._MOCK_DONE, )

    class _BrainParameters:

        def __init__(self):
            self.vector_action_space_size = MockUnityEnvironmentVisual._MOCK_ACTION_SIZE
            self.vector_action_space_type = MockUnityEnvironmentVisual._MOCK_ACTION_TYPE
            self.vector_action_descriptions = MockUnityEnvironmentVisual._MOCK_ACTION_DESCRIPTIONS

            self.camera_resolutions = MockUnityEnvironmentVisual._MOCK_STATE.shape

    def __init__(self):
        self.brain_names = self._BRAIN_NAMES
        self.brains = {self.brain_names[0]: self._BrainParameters(), }

    def reset(self, train_mode=True):
        self._train_mode = train_mode
        return {self.brain_names[0]: self._BrainInfo(), }

    def step(self, action):
        self._action = action
        return {self.brain_names[0]: self._BrainInfo(), }
