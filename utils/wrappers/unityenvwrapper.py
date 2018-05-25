from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from humblerl import Environment
from humblerl.utils import doc_inherit


class UnityEnvWrapper(Environment):
    """Wrapper on UnityEnvironment implementing Environment interface.

    It supports only single agent environments.
    """

    def __init__(self, unity_env, use_observations=False):
        """Initialize UnityEnvWrapper.

        Initialize UnityEnvWrapper with injected UnityEnvironment.

        Args:
            unity_env (UnityEnvironment): Unity environment to wrap.
            use_observations (bool): If visual observations (instead of vector observations) should
        be used. Currently only observations from one camera are supported. (default: False)
        """

        super(UnityEnvWrapper, self).__init__()

        self._env = unity_env
        self._default_brain = self._env.brain_names[0]
        self._use_observations = use_observations

        brain_parameters = self._env.brains[self._default_brain]

        self._action_space_info = Environment.ActionSpaceInfo(
            size=brain_parameters.vector_action_space_size,
            type=brain_parameters.vector_action_space_type,
            descriptions=brain_parameters.vector_action_descriptions
        )

        if use_observations:
            self._state_space_info = Environment.StateSpaceInfo(
                size=brain_parameters.camera_resolutions,
                type="observation"
            )
        else:
            self._state_space_info = Environment.StateSpaceInfo(
                size=brain_parameters.vector_observation_space_size,
                type=brain_parameters.vector_observation_space_type
            )

    @doc_inherit
    def _reset(self, train_mode=True):
        brain_info = self._env.reset(train_mode=train_mode)[
            self._default_brain]

        if self._use_observations:
            # Layout of visual observations is HWC
            state = brain_info.visual_observations[0][0, :, :, :]
        else:
            state = brain_info.vector_observations[0]

        return state

    @doc_inherit
    def _step(self, action):
        assert action is not None

        # Take brain info of first (default) brain
        brain_info = self._env.step(action)[self._default_brain]

        # Take state, reward and done flag of first agent
        if self._use_observations:
            state = brain_info.visual_observations[0][0, :, :, :]
        else:
            state = brain_info.vector_observations[0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return (state, reward, done)
