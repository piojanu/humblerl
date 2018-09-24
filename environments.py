import gym
import gym_sokoban
import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    @abstractmethod
    def render(self):
        """Show/print some visual representation of environment's current state."""

        pass

    @abstractmethod
    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation mode.
                (Default: True)

        Returns:
            object: The initial state. 
        """

        pass

    @abstractmethod
    def step(self, action):
        """Perform action in environment.

        Args:
            action (object): Action to perform.

        Returns:
            object: New state.
            float: Reward.
            bool: Flag indicating if episode has finished.
            object: Environment diagnostic information if available, otherwise None.
        """

        pass

    @abstractproperty
    def action_space(self):
        """Get action space definition.

        Returns:
            object: Action space representation depends on environment.
        """

        pass

    @abstractproperty
    def state_space(self):
        """Get environment state space definition.

        Returns:
            object: State space representation depends on environment.
        """

        pass

    @abstractproperty
    def current_state(self):
        """Get current environment's observable state.

        Returns:
            object: Current state.
        """

        pass

    @abstractproperty
    def valid_actions(self):
        """Get currently available actions.

        Returns:
            object: Available actions representation depends on environment.
        """

        pass


class MDP(metaclass=ABCMeta):
    """Interface for MDP, describes state and action spaces and their dynamics."""

    @abstractmethod
    def simulate(self, state, action):
        """Perform `action` in `state`. Return outcome.

        Args:
            state (object): MDP's state.
            action (object): MDP's action.

        Returns:
            object: New state.
            float: Reward.
            bool: Flag indicating if episode has ended.
        """

        pass

    @abstractmethod
    def get_init_state(self):
        """Prepare and return initial state.

        Returns:
            object: Initial state.
        """

        pass

    @abstractmethod
    def get_valid_actions(self, state):
        """Get available actions in `state`.

        Args:
            state (object): MDP's state.

        Returns:
            object: Available actions representation depends on environment.
        """

        pass

    @abstractproperty
    def action_space(self):
        """Get action space definition.

        Returns:
            object: Action space representation depends on model.
        """

        pass

    @abstractproperty
    def state_space(self):
        """Get environment state space definition.

        Returns:
            object: State space representation depends on model.
        """

        pass
