import gym
import gym_sokoban
import numpy as np

from .core import Environment


class GymEnvironment(Environment):
    """Wrapper on OpenAI Gym toolkit environments."""

    def __init__(self, env):
        "Initialize OpenAI Gym wrapper"

        self.env = env
        self._players_number = 1

        # Get state space
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Discrete):
            self._state_space = np.array([obs_space.n])
        elif isinstance(obs_space, gym.spaces.Box):
            self._state_space = np.concatenate((
                np.expand_dims(obs_space.low, axis=-1),
                np.expand_dims(obs_space.high, axis=-1)), axis=-1)
        else:
            raise ValueError(
                "For OpenAI Gym only discrete and box state spaces are supported")

        # Get action space
        act_space = self.env.action_space
        if isinstance(act_space, gym.spaces.Discrete):
            self._valid_actions = np.array(list(range(act_space.n)))
        else:
            raise ValueError(
                "For OpenAI Gym only discrete action space is supported")

    def reset(self, train_mode=True, first_player=0):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
                mode. (Default: True)
            first_player (int): Index of player who starts game. (Default: 0)

        Returns:
            np.array: The initial state. 
            int: Current player (first is 0).

        Note:
            `train_mode` and `first_player` arguments are ignored in OpenAI Gym.
        """

        self._current_state = self.env.reset()
        return self._current_state, 0

    def step(self, action):
        """Perform action in environment.

        Args:
            action (np.array): Action to perform. In discrete action space it's single
                item with action number. In continuous case, it's action vector.

        Returns:
            np.array: New state.
            int: Next player (first is 0).
            float: Reward.
            bool: Flag indicating if episode has ended.
            object: Environment diagnostic information if available otherwise None.
        """

        # For now we only support discrete action spaces in OpenAI Gym
        assert not isinstance(action, np.ndarray), \
            "For OpenAI Gym only discrete action space is supported"

        self._current_state, reward, done, info = self.env.step(action)
        return self._current_state, 0, reward, done, info

    def render(self):
        """Show/print some visual representation of environment's current state."""

        self.env.render()


def create_gym(env_name):
    """Create OpenAI Gym environment with given name.

    Args:
        env_name (string): Environment name passed to `gym.make(...)` function.

    Returns:
        GymEnvironment: OpenAI Gym environment wrapped in `hrl.Environment` interface.
    """

    return GymEnvironment(gym.make(env_name))
