from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import pytest

from humblerl.environments import Environment


class MockSpecificEnvironment(Environment):
    _MOCK_INIT_STATE = [1, 2, 3]
    _MOCK_NEXT_STATE = [4, 5, 6]
    _MOCK_REWARD = 7.
    _MOCK_DONE = False

    def _reset(self, train_mode):
        return self._MOCK_INIT_STATE

    def _step(self, action):
        return (self._MOCK_NEXT_STATE, self._MOCK_REWARD, self._MOCK_DONE)


class TestEnvironment(object):
    _MOCK_ACTION = [7., ]

    @pytest.fixture
    def specific_env_mock(self):
        return MockSpecificEnvironment()

    def test_reset(self, specific_env_mock):
        state = specific_env_mock.reset()

        assert state == MockSpecificEnvironment._MOCK_INIT_STATE
        assert specific_env_mock.current_state == MockSpecificEnvironment._MOCK_INIT_STATE

    def test_step(self, specific_env_mock):
        state, reward, done = specific_env_mock.step(self._MOCK_ACTION)

        assert state == MockSpecificEnvironment._MOCK_NEXT_STATE
        assert reward == MockSpecificEnvironment._MOCK_REWARD
        assert done == MockSpecificEnvironment._MOCK_DONE
        assert specific_env_mock.current_state == MockSpecificEnvironment._MOCK_NEXT_STATE

    def test_reset_step(self, specific_env_mock):
        init = specific_env_mock.reset()
        state, reward, done = specific_env_mock.step(self._MOCK_ACTION)

        assert init == MockSpecificEnvironment._MOCK_INIT_STATE
        assert state == MockSpecificEnvironment._MOCK_NEXT_STATE
        assert reward == MockSpecificEnvironment._MOCK_REWARD
        assert done == MockSpecificEnvironment._MOCK_DONE
        assert specific_env_mock.current_state == MockSpecificEnvironment._MOCK_NEXT_STATE
