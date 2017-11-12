from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from humblerl.environments import Environment
from humblerl.utils import doc_inherit

from unityagents import UnityEnvironment


class UnityEnvWrapper(Environment):
    """Wrapper on UnityEnvironment implementing Environment interface.

    It supports only single agent environments.
    """

    def __init__(self, file_name=None, unity_env=None, UnityEnvironmentType=UnityEnvironment):
        """Initialize UnityEnvWrapper.

        Initialize UnityEnvWrapper with injected UnityEnvironment or create new
        UnityEnvironment based on path in file_name parameter.

        Args:
            file_name (string): Path to Unity3D environment. (default: None)
            unity_env (UnityEnvironment): Unity environment. (default: None)
            UnityEnvironmentType (UnityEnvironment): Enables dependency injection.
        """

        if file_name is not None and unity_env is None:
            self._env = UnityEnvironmentType(file_name=file_name)
        elif unity_env is not None and file_name is None:
            self._env = unity_env
        else:
            raise ValueError("Illegal initialization path!")

        self._default_brain = self._env.brain_names[0]

    @doc_inherit
    def reset(self, train_mode=True, context=None):
        brain_info = self._env.reset(train_mode=train_mode)[
            self._default_brain]
        return brain_info.states[0]

    @doc_inherit
    def step(self, action, context=None):
        assert action is not None

        # Take brain info of first (default) brain
        brain_info = self._env.step(action)[self._default_brain]

        # Take state, reward and done flag of first agent
        state = brain_info.states[0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return (state, reward, done)
