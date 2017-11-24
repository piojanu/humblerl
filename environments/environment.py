from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from collections import namedtuple


class Environment(object):
    """Interface for environments."""

    ActionSpaceInfo = namedtuple(
        "ActionSpaceInfo", ["size", "type", "descriptions"])
    StateSpaceInfo = namedtuple(
        "StateSpaceInfo", ["size", "type"])

    DISCRETE_SPACE = "discrete"
    CONTINUOUS_SPACE = "continuous"

    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. (default: True)

        Returns:
            np.array: The initial state. 
        """

        raise NotImplementedError()

    def step(self, action):
        """Perform action in environment and return new state, reward and done flag.

        Args:
            action (list of floats): Action to perform. In discrete action space
        it's single element list with action number.

        Returns:
            np.array: New state.
            float: Next reward.
            bool: Flag indicating if episode has ended.
        """

        raise NotImplementedError()

    @property
    def action_space_info(self):
        """Properties of action space.

        Returns:
            Environment.ActionSpaceInfo: Include size and type (DISCRETE_SPACE 
        or CONTINUOUS_SPACE) of action space. Also descriptions of actions.
        """

        return self._action_space_info

    @property
    def state_space_info(self):
        """Properties of state space.

        Returns:
            Environment.StateSpaceInfo: Include size and type (DISCRETE_SPACE 
        or CONTINUOUS_SPACE) of state space.
        """

        return self._state_space_info
