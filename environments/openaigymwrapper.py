from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from humblerl.environments import Environment
from humblerl.utils import doc_inherit

import gym


class OpenAIGymWrapper(Environment):
    """Wrapper on Open AI Gym implementing Environment interface."""

    def __init__(self, gym_env):
        """Initialize UnityEnvWrapper.

        Initialize UnityEnvWrapper with injected UnityEnvironment.

        Args:
            unity_env (UnityEnvironment): Unity environment to wrap.
        """

        super(OpenAIGymWrapper, self).__init__()

        self._env = gym_env

        if type(self._env.action_space) is gym.spaces.box.Box:
            self._action_space_info = Environment.ActionSpaceInfo(
                size=self._env.action_space.shape,
                type="continuous",
                descriptions=repr(self._env.action_space)
            )
        elif type(self._env.action_space is gym.spaces.discrete.Discrete):
            self._action_space_info = Environment.ActionSpaceInfo(
                size=self._env.action_space.n,
                type="discrete",
                descriptions=repr(self._env.action_space)
            )
        else:
            raise ValueError("Unknown action space type!")

        if type(self._env.observation_space) is gym.spaces.box.Box:
            self._state_space_info = Environment.StateSpaceInfo(
                size=self._env.observation_space.shape,
                type="continuous"
            )
        elif type(self._env.observation_space is gym.spaces.discrete.Discrete):
            self._state_space_info = Environment.StateSpaceInfo(
                size=self._env.observation_space.n,
                type="discrete",
            )
        else:
            raise ValueError("Unknown observation space type!")

    @doc_inherit
    def _reset(self, train_mode):
        self._train_mode = train_mode

        return self._env.reset()

    @doc_inherit
    def _step(self, action):
        assert action is not None

        # Take brain info of first (default) brain
        state, reward, done, _ = self._env.step(action)

        if not self._train_mode:
            self._env.render()

        return (state, reward, done)
