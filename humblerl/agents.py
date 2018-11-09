import numpy as np

from abc import ABCMeta, abstractmethod


class Mind(metaclass=ABCMeta):
    """Artificial mind of RL agent."""

    @abstractmethod
    def plan(self, state, train_mode, debug_mode):
        """Do forward pass through agent model (inference/planning) on state.

        Args:
            state (object): State of environment to inference on.
            train_mode (bool): Informs planner whether it's in training or evaluation mode.
                E.g. in evaluation it can optimise graph, disable exploration etc.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            np.array: Actions scores (e.g. unnormalized log probabilities/Q-values/etc.)
                possibly raw Artificial Neural Net output i.e. logits.
            object (optional): Mind's extra information, passed to 'on_action_planned' callback.
                If you will omit it, it will be set to None by default.
        """

        pass


class Vision(object):
    """Vision system entity in Reinforcement Learning task.

       It is responsible for e.g. data preprocessing, feature extraction etc.
    """

    def __init__(self, state_processor_fn=None, reward_processor_fn=None):
        """Initialize vision processors.

        Args:
            state_processor_fn (callable): Function for state processing. It should
                take raw environment state as an input and return processed state.
                (Default: None which will result in passing raw state)
            reward_processor_fn (callable): Function for reward processing. It should
                take raw environment reward as an input and return processed reward.
                (Default: None which will result in passing raw reward)
        """

        self._process_state = \
            state_processor_fn if state_processor_fn is not None else lambda x: x
        self._process_reward = \
            reward_processor_fn if reward_processor_fn is not None else lambda x: x

    def __call__(self, state, reward=0.):
        return self._process_state(state), self._process_reward(reward)


class ChainVision:
    """Chain many vision objects.

    Args:
        *args: Vision objects to chain calls in order.
    """

    def __init__(self, *args):
        self.visions = args

    def __call__(self, state, reward=0.):
        for vision in self.visions:
            state, reward = vision(state, reward)
        return state, reward


class RandomAgent(Mind):
    """Mind that acts at random (uniformly)."""

    def __init__(self, env):
        """Initialize random agent.

        Args:
            env (Environment): Game's environment.
        """

        self.env = env

    def plan(self, state, train_mode, debug_mode):
        """Ignores all arguments and return random action from action space.

        Args:
            ...see `hrl.Environment::valid_actions` docstring...

        Returns:
            np.ndarray: Action vector (one-hot for discrete envs).
        """

        action = self.env.sample_action()
        if self.env.is_discrete:
            one_hot = np.zeros(self.env.action_space.num)
            one_hot[action] = 1
            action = one_hot
        return action
