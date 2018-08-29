import gym
import gym_sokoban
import numpy as np

from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    @abstractmethod
    def render(self):
        """Show/print some visual representation of environment's current state."""

        pass

    @abstractmethod
    def reset(self, train_mode=True, first_player=0):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
                mode. (Default: True)
            first_player (int): Index of player who starts game. (Default: 0)

        Returns:
            np.array: The initial state. 
            int: Current player (first is 0).
        """

        pass

    @abstractmethod
    def step(self, action):
        """Perform action in environment.

        Args:
            action (int or np.array): Action to perform. In discrete action space it's integer
                action number. In continuous case, it's action vector (np.array).

        Returns:
            np.array: New state.
            int: Next player (first is 0).
            float: Reward.
            bool: Flag indicating if episode has ended.
            object: Environment diagnostic information if available otherwise None.
        """

        pass

    @property
    def current_player(self):
        """Access current player index in environment state.

        Returns:
            int: Current player (first is 0).

        Note:
            In child class just set self._current_player
        """

        return self._current_player

    @property
    def current_state(self):
        """Access current observable state in which environment is.

        Returns:
            np.array: Current observable environment state.

        Note:
            In child class just set self._current_state
        """

        return self._current_state

    @property
    def players_number(self):
        """Access number of players that take actions in this MDP.

        Returns:
            int: Number of players (first is 0).

        Note:
            In child class just set `self._players_number`.
        """

        return self._players_number

    @property
    def action_space(self):
        """Access currently (this state) valid actions.

        Returns:
            int: It's integer describing action space size.

        Note:
            In child class just set `self._action_space`.
            For now only discrete actions are supported!
        """

        return self._action_space

    @property
    def state_space(self):
        """Access environment state space.

        Returns:
            int or np.array: If desecrate state space, then it's integer describing state space size.
                If continuous, then this is (M + 1) dimensional array, where first M dimensions are
                state dimensions and last dimension of size 2 keeps respectively [min, max]
                (inclusive range) values which given state feature can take.

        Note:
            In child class just set `self._state_space`.
        """

        return self._state_space

    @property
    def valid_actions(self):
        """Access currently (this state) valid actions.

        Returns:
            np.array: It's a 1D array with available action values.

        Note:
            In child class just set `self._valid_actions`.
            For now only discrete actions are supported!
        """

        return self._valid_actions


class Model(metaclass=ABCMeta):
    """Represents some MDP, describes state and action spaces and give access to dynamics."""

    @abstractmethod
    def simulate(self, state, player, action):
        """Perform `action` as `player` in `state`. Return outcome.

        Args:
            state (np.array): State of MDP.
            player (int): Current player index.
            action (np.array): Action to perform. In discrete action space it's single
                item with action number. In continuous case, it's action vector.

        Returns:
            np.array: New state.
            int: Next player (first is 0).
            float: Reward.
            bool: Flag indicating if episode has ended.
        """

        pass

    @property
    @abstractmethod
    def action_space(self, state):
        """Access valid actions of given MDP state.

        Args:
            state (np.array): State of MDP.

        Returns:
            np.array: If desecrate action space, then it's a 1D array with available action values.
                If continuous, then this is 2D array, where first dimension has action vector size
                and second dimension of size 2 keeps respectively [min, max] (inclusive range)
                values which given action vector element can take.
        """

        pass

    @property
    @abstractmethod
    def players_number(self):
        """Access number of players that take actions in this MDP.

        Returns:
            int: Number of players (first is 0).
        """

        pass

    @property
    @abstractmethod
    def state_space(self):
        """Access environment state space.

        Returns:
            int or np.array: If desecrate state space, then it's integer describing state space size.
                If continuous, then this is (M + 1)-D array, where first M dimensions are
                state dimensions and last dimension of size 2 keeps respectively [min, max]
                (inclusive range) values which given state feature can take.
        """

        pass


class GymEnvironment(Environment):
    """Wrapper on OpenAI Gym toolkit environments."""

    def __init__(self, env):
        "Initialize OpenAI Gym wrapper"

        self.env = env
        self._players_number = 1
        self._current_player = 0

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
            self._action_space = act_space.n
        else:
            raise ValueError("For OpenAI Gym only discrete action space is supported")

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
