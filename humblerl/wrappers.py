import gym
import gym_maze
import gym_sokoban
import numpy as np

from .environments import Environment, Discrete, Continuous


class GymEnvironment(Environment):
    """Wrapper on OpenAI Gym toolkit environments."""

    def __init__(self, env):
        """Initialize OpenAI Gym wrapper"""

        self.env = env
        self.step_counter = 0

        # Get state space
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Discrete):
            self._state_space = obs_space.n
        elif isinstance(obs_space, gym.spaces.Box):
            self._state_space = np.concatenate((
                np.expand_dims(obs_space.low, axis=-1),
                np.expand_dims(obs_space.high, axis=-1)), axis=-1)
        else:
            raise ValueError("For OpenAI Gym only discrete and box state spaces are supported")

        # Get action space
        act_space = self.env.action_space
        if isinstance(act_space, gym.spaces.Discrete):
            self._valid_actions = np.array(list(range(act_space.n)))
            self._action_space = Discrete(num=act_space.n)
        else:
            n_params = len(act_space.low)
            self._action_space = Continuous(num=n_params, low=act_space.low, high=act_space.high)
            self._valid_actions = self._action_space

    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation mode.
                (Default: True)

        Returns:
            np.ndarray: The initial state.

        Note:
            `train_mode` argument is ignored in OpenAI Gym.
        """

        self._current_state = self.env.reset()
        self.step_counter = 0
        return self._current_state

    def step(self, action):
        """Perform action in environment.

        Args:
            action (int or np.ndarray): Action to perform. In discrete action space it's action
                index. In continuous case, it's action vector.

        Returns:
            object: New state.
            float: Reward.
            bool: Flag indicating if episode has finished.
            object: Environment diagnostic information if available, otherwise None.
        """

        self._current_state, reward, done, info = self.env.step(action)
        self.step_counter += 1
        return self._current_state, reward, done, info

    def render(self):
        """Show/print some visual representation of environment's current state."""

        self.env.render()

    @property
    def action_space(self):
        """Get action space definition.

        Returns:
            ActionSpace: Action space, discrete or continuous.
        """

        return self._action_space

    @property
    def state_space(self):
        """Get environment state space definition.

        Returns:
            int or np.ndarray: If desecrate state space, then it's integer describing state space size.
                If continuous, then this is (M + 1) dimensional array, where first M dimensions are
                state dimensions and last dimension of size 2 keeps respectively [min, max]
                (inclusive range) values which given state feature can take.
        """

        return self._state_space

    @property
    def current_state(self):
        """Get current environment's observable state.

        Returns:
            np.ndarray: Current state.
        """

        return self._current_state

    @property
    def valid_actions(self):
        """Get currently available actions.

        Returns:
            np.ndarray/Continuous: Discrete env: np.ndarray with enumerated valid actions
                for current state. Continous env: Valid continuous action space for current state.
        """

        return self._valid_actions


def create_gym(env_name):
    """Create OpenAI Gym environment with given name.

    Args:
        env_name (string): Environment name passed to `gym.make(...)` function.

    Returns:
        GymEnvironment: OpenAI Gym environment wrapped in `hrl.Environment` interface.
    """

    return GymEnvironment(gym.make(env_name))
