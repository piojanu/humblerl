from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from abc import ABCMeta, abstractmethod
from humblerl import Transition

class Policy(metaclass=ABCMeta):
    """Abstract class representing policy in Reinforcement Learning task."""

    @abstractmethod
    def select_action(self, curr_state):
        """Evaluate policy and return action.

        Returns:
            list of floats: action to take in the environment.
        """
        pass

    @abstractmethod
    def report(self, transition):
        """Inform policy about transition in the environment.

        Args:
            transition (Transition): Transition packed in namedtuple: 
        state, action, reward, next_state, is_terminal.
        """
        pass


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

    def __init__(self, env, model, vision=Vision()):
        """Initialize agent object.

        Args:
            env (Environment): Any environment implementing
        humblerl.environments.Environment interface.
            model (Policy): Any model implementing Policy
        interface.
            vision (Vision): Processes raw environment output
        before passing it to the agent. [Default: Vision()]
        """

        self._env = env
        self._model = model
        self._vision = vision

    @property
    def environment(self):
        """Access environment."""
        return self._env

    @property
    def model(self):
        """Access model."""
        return self._model

    @property
    def vision(self):
        """Access vision."""
        return self._vision

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
        it's single element list with action number. If None, agent will query its model
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
            action = self.model.select_action(curr_state=curr_state)

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

        # Inform model about transition
        self.model.report(transition=transition)

        return transition

    def play(self, max_steps=-1):
        """Python generator. Play agent in environment.

        Args:
            max_steps (int): Play until env is done or max_steps is reached.
        If -1, then play until env is done. [Default: -1]

        Returns:
            transition (Transition): Transition packed in namedtuple: 
        state, action, reward, next_state, is_terminal.
        """

        stop = False
        step = 0

        while not stop and (max_steps == -1 or step < max_steps):
            transition = self.do()
            stop = transition.is_terminal

            yield transition
            step += 1
