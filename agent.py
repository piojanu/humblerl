from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from abc import ABCMeta, abstractmethod
from humblerl import Transition


class Vision(object):
    """Vision system entity in Reinforcement Learning task."""

    def __init__(self, state_processor=None, reward_processor=None):
        """Initialize vision processors.

        Args:
            state_processor (function): Function for state processing. It should
        take raw environment state as an input and return processed state.
        [Default: None which will result in passing raw state]
            reward_processor (function): Function for reward processing. It should
        take raw environment reward as an input and return processed reward.
        [Default: None which will result in passing raw reward]
        """

        self._process_state = \
            state_processor if state_processor is not None else lambda x: x
        self._process_reward = \
            reward_processor if reward_processor is not None else lambda x: x

    def __call__(self, state, reward=None):
        if reward == None:
            return self._process_state(state)
        else:
            return self._process_state(state), self._process_reward(reward)


class Agent(object):
    """Agent entity in Reinforcement Learning task."""

    def __init__(self, env, policy, vision=Vision(), callbacks=[]):
        """Initialize agent object.

        Args:
            env (Environment): Any environment implementing
        humblerl.environments.Environment interface.
            policy (Policy): Any policy implementing Policy
        interface.
            vision (Vision): Processes raw environment output
        before passing it to the agent. [Default: Vision()]
        """

        self._env = env
        self._policy = policy
        self._vision = vision
        self._callbacks = callbacks

    @property
    def environment(self):
        """Access environment."""
        return self._env

    @property
    def policy(self):
        """Access policy."""
        return self._policy

    @property
    def vision(self):
        """Access vision."""
        return self._vision

    @property
    def callbacks(self):
        """Access callbacks."""
        return self._callbacks

    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. [Default: True]

        Returns:
            np.array: The initial state. 
        """

        # Reset environment.
        raw_state = self.environment.reset(train_mode=train_mode)

        # Process raw state
        state = self.vision(raw_state)
        self._cur_state = state

        return state

    def do(self, action=None):
        """Take a step in the environment and process output with vision system.

        Args:
            action (list of floats): Action to perform. In discrete action space
        it's single element list with action number. If None, agent will query its policy
        for action. [Default: None]

        Returns:
            transition (Transition): Transition packed in namedtuple: 
        state, action, reward, next_state, is_terminal.
        """

        # Checks if everything needed to take a step is present
        if self.environment.current_state is None:
            raise ValueError("You need to reset agent first!")

        curr_state = self.vision(self.environment.current_state)

        # Get next action
        if action == None:
            action = self.policy(state=curr_state)

        # Take a step in environment
        raw_state, raw_reward, done = self.environment.step(action=action)

        # Process raw state and reward with vision system
        state, reward = self.vision(raw_state, raw_reward)

        # Collect transition
        transition = Transition(
            state=curr_state,
            action=action,
            reward=reward,
            next_state=state,
            is_terminal=done
        )

        # Inform callbacks about transition
        for callback in self._callbacks:
            callback.report_step(transition=transition)

        return transition

    def play(self, max_steps=-1):
        """Python generator. Play agent in environment.

        Args:
            max_steps (int): Play until env is done or max_steps is reached.
        If -1, then play until env is done. [Default: -1]

        Returns:
            Nothing
        """

        stop = False
        step = 0

        while not stop and (max_steps == -1 or step < max_steps):
            transition = self.do()
            stop = transition.is_terminal

            step += 1
