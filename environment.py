from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from abc import ABCMeta, abstractmethod
from collections import namedtuple

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "is_terminal"])


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    ActionSpaceInfo = namedtuple(
        "ActionSpaceInfo", ["size", "type", "descriptions"])
    StateSpaceInfo = namedtuple(
        "StateSpaceInfo", ["size", "type"])

    DISCRETE_SPACE = "discrete"
    CONTINUOUS_SPACE = "continuous"

    def __init__(self):
        """Initialize Environment object."""

        self._curr_state = None

    @abstractmethod
    def _reset(self, train_mode):
        """This function should be implemented in derived classes.

        Interface is the same as in Environment.reset(...)."""
        pass

    @abstractmethod
    def _step(self, action):
        """This function should be implemented in derived classes.

        Interface is the same as in Environment.step(...)."""
        pass

    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. (default: True)

        Returns:
            np.array: The initial state. 
        """

        self._curr_state = self._reset(train_mode)

        return self._curr_state

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

        self._curr_state, reward, done = self._step(action)

        return (self._curr_state, reward, done)

    @property
    def current_state(self):
        """Access state.

        Returns:
            np.array: Current environment state.
        """

        return self._curr_state

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
