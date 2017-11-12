import numpy as np
import pytest

from humblerl.environments import UnityEnvWrapper


class MockUnityEnvironment(object):
    _MOCK_STATE = np.ones((3, 3), dtype=float)
    _MOCK_REWARD = 0.
    _MOCK_DONE = False
    _BRAIN_NAMES = ("DEFAULT_BRAIN", )

    class _BrainInfo:

        def __init__(self):
            self.states = (MockUnityEnvironment._MOCK_STATE, )
            self.rewards = (MockUnityEnvironment._MOCK_REWARD, )
            self.local_done = (MockUnityEnvironment._MOCK_DONE, )

    def __init__(self, file_name):
        self._file_name = file_name

        self.brain_names = self._BRAIN_NAMES

    def reset(self, train_mode=True):
        self._train_mode = train_mode
        return {self.brain_names[0]: self._BrainInfo(), }

    def step(self, action):
        self._action = action
        return {self.brain_names[0]: self._BrainInfo(), }


class TestUnityEnvWrapperInit(object):
    def test_init_unityenvwrapper_with_path(self):
        FILE_NAME = "./test"

        env = UnityEnvWrapper(file_name=FILE_NAME,
                              UnityEnvironmentType=MockUnityEnvironment)

        assert env._env._file_name == FILE_NAME
        assert env._default_brain == MockUnityEnvironment._BRAIN_NAMES[0]

    def test_init_unityenvwrapper_with_unityenv(self):
        MOCK_ENV = MockUnityEnvironment(None)

        env = UnityEnvWrapper(unity_env=MOCK_ENV)

        assert env._env is MOCK_ENV
        assert env._default_brain == MockUnityEnvironment._BRAIN_NAMES[0]

    @pytest.fixture(params=[{"file_name": None, "unity_env": None},
                            {"file_name": "test", "unity_env": MockUnityEnvironment(None)}])
    def illegall_init_params(self, request):
        return request.param

    def test_illegally_init_unityenvwrapper(self, illegall_init_params):
        with pytest.raises(ValueError):
            UnityEnvWrapper(file_name=illegall_init_params["file_name"],
                            unity_env=illegall_init_params["unity_env"])


class TestUnityEnvWrapperRun(object):
    @pytest.fixture()
    def unityenvwrapper(self):
        return UnityEnvWrapper(unity_env=MockUnityEnvironment(None))

    def test_reset(self, unityenvwrapper):
        TRAIN_MODE = False

        state = unityenvwrapper.reset(train_mode=TRAIN_MODE)

        assert unityenvwrapper._env._train_mode == TRAIN_MODE
        assert np.array_equal(state, MockUnityEnvironment._MOCK_STATE)

    def test_step(self, unityenvwrapper):
        ACTION = (0, 1, 2)

        state, reward, done = unityenvwrapper.step(ACTION)

        assert unityenvwrapper._env._action == ACTION
        assert np.array_equal(state, MockUnityEnvironment._MOCK_STATE)
        assert reward == MockUnityEnvironment._MOCK_REWARD
        assert done == MockUnityEnvironment._MOCK_DONE
