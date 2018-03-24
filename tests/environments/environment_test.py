from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pytest

from humblerl.environments import UnityEnvWrapper
from mockunityenvironment import MockUnityEnvironmentVector, MockUnityEnvironmentVisual


class TestEnvironmentWrapper(object):
    @pytest.fixture(params=[
        # Test wrapper for Unity environment with vector observations (state)
        UnityEnvWrapper(unity_env=MockUnityEnvironmentVector(), use_observations=False),
        # Test wrapper for Unity environment with visual observations (image)
        UnityEnvWrapper(unity_env=MockUnityEnvironmentVisual(), use_observations=True),
    ], ids=["UnityEnv vector observations", "UnityEnv visual observations"])
    def envwrapper(self, request):
        wrapper, mockenv = request.param, request.param._env
        return wrapper, mockenv

    def test_reset(self, envwrapper):
        wrapper, mockenv = envwrapper
        TRAIN_MODE = False

        state = wrapper.reset(train_mode=TRAIN_MODE)

        assert mockenv._train_mode == TRAIN_MODE
        assert np.array_equal(state, mockenv._MOCK_STATE)

    def test_step(self, envwrapper):
        wrapper, mockenv = envwrapper
        ACTION = (0, 1, 2)

        state, reward, done = wrapper.step(ACTION)

        assert mockenv._action == ACTION
        assert np.array_equal(state, mockenv._MOCK_STATE)
        assert reward == mockenv._MOCK_REWARD
        assert done == mockenv._MOCK_DONE

    def test_action_space_info(self, envwrapper):
        wrapper, mockenv = envwrapper

        assert wrapper.action_space_info.size == \
            mockenv._MOCK_ACTION_SIZE
        assert wrapper.action_space_info.type == \
            mockenv._MOCK_ACTION_TYPE
        assert wrapper.action_space_info.descriptions == \
            mockenv._MOCK_ACTION_DESCRIPTIONS

    def test_state_space_info(self, envwrapper):
        wrapper, mockenv = envwrapper

        assert wrapper.state_space_info.size == \
            mockenv._MOCK_STATE.shape
        assert wrapper.state_space_info.type == \
            mockenv._MOCK_STATE_TYPE
