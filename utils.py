import numpy as np

from . import Mind


class RandomAgent(Mind):
    """Mind that acts at random (uniformly)."""

    def __init__(self, action_space):
        """Initialize random agent.

        Args:
            action_space (np.ndarray): Discrete or continuous hrl.Environment action_space. 
        """

        self.action_space = action_space

    def plan(self, state, player, train_mode, debug_mode):
        """Ignores all arguments and return random action from action space.

        Args:
            ...see `hrl.Environment::valid_actions` docstring...

        Returns:
            int or np.ndarray: Random action from action space.
        """

        if len(self.action_space.shape) == 1:
            one_hot = np.zeros_like(self.action_space)
            one_hot[np.random.choice(self.action_space)] = 1
            return one_hot
        elif len(self.action_space.shape) == 2:
            return np.random.uniform(self.action_space.T[0], self.action_space.T[1])
        else:
            raise ValueError("Invalid action space!")
