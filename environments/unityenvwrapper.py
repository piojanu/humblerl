from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from humblerl.environments import Environment
from humblerl.utils import doc_inherit


class UnityEnvWrapper(Environment):
    """Wrapper on UnityEnvironment implementing Environment interface.

    It supports only single agent environments.
    """

    def __init__(self, unity_env):
        """Initialize UnityEnvWrapper.

        Initialize UnityEnvWrapper with injected UnityEnvironment.

        Args:
            unity_env (UnityEnvironment): Unity environment to wrap.
        """

        self._env = unity_env
        self._default_brain = self._env.brain_names[0]

        brain_parameters = self._env.brains[self._default_brain]

        self._action_space_info = Environment.ActionSpaceInfo(
            size=brain_parameters.action_space_size,
            type=brain_parameters.action_space_type,
            descriptions=brain_parameters.action_descriptions
        )

        self._state_space_info = Environment.StateSpaceInfo(
            size=brain_parameters.state_space_size,
            type=brain_parameters.state_space_type
        )

    @doc_inherit
    def reset(self, train_mode=True):
        brain_info = self._env.reset(train_mode=train_mode)[
            self._default_brain]
        return brain_info.states[0]

    @doc_inherit
    def step(self, action):
        assert action is not None

        # Take brain info of first (default) brain
        brain_info = self._env.step(action)[self._default_brain]

        # Take state, reward and done flag of first agent
        state = brain_info.states[0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return (state, reward, done)
