from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from humblerl.environments import Transition


class Context(object):
    """Context carry useful information and objects for training."""

    def __init__(self, logger=None, policy_info=None):
        """Initialize context.

        Args:
            logger (utils.Logger): Object that allows for gathering statistics.
            policy_info (object): User defined object returned from Agent's policy.
        """

        self.logger = logger
        self.policy_info = policy_info


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

    def __call__(self, state, reward):
        return self._process_state(state), self._process_reward(reward)


class Agent(object):
    """Agent entity in Reinforcement Learning task."""

    def __init__(self, env, vision=Vision()):
        """Initialize agent object.

        Args:
            env (humblerl.environments.Environment): Any environment implementing
        HumbleRL Environment interface.
            vision (humblerl.agents.Vision): Processes raw environment output
        before passing it to the agent. [Default: humblerl.agents.Vision()]
        """

        self._env = env
        self._vision = vision

        self._cur_policy = None
        self._cur_state = None

    @property
    def environment(self):
        """Access environment."""
        return self._env

    @property
    def policy(self):
        """Access current policy."""
        return self._cur_policy

    @policy.setter
    def policy(self, value):
        """Set current policy."""
        self._cur_policy = value

    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. [Default: True]

        Returns:
            np.array: The initial state. 
        """

        # Reset environment.
        raw_state = self._env.reset(train_mode=train_mode)

        # Process raw state
        state, _ = self._vision(raw_state, 0)
        self._cur_state = state

        return state

    def step(self, policy=None):
        """Take a step in the environment and process output with vision system.

        Args:
            policy (function): Function that takes state/observation and return
        action (list of floats) to take in the environment and user info (optional).
        In discrete action space it's single element list with action number.
        If None, previous policy will be used (it's called current policy).
        [Default: None]

        Returns:
            transition (environments.Transition): Transition packed in namedtuple: 
        state, action, reward, next_state, is_terminal.
            context (agents.Context): Includes user defined object,
        returned from policy.
        """

        # Assign new policy if given
        if policy is not None:
            self._cur_policy = policy

        # Checks if everything needed to take a step is present
        if self._cur_state is None:
            raise ValueError("You need to reset agent first!")

        if self._cur_policy is None:
            raise ValueError("You need to provide agent policy!")

        # Get next action and possible user defined info
        policy_return = self._cur_policy(self._cur_state)
        action, info = None, None
        if isinstance(policy_return, tuple):
            # When policy returns user defined info too
            action, info = policy_return
        else:
            # When policy returns next action only
            action = policy_return

        # Take a step in environment
        raw_state, raw_reward, done = self._env.step(action=action)

        # Process raw state and reward with vision system
        state, reward = self._vision(raw_state, raw_reward)

        # Collect transition
        transition = Transition(
            state=self._cur_state,
            action=action,
            reward=reward,
            next_state=state,
            is_terminal=done
        )

        self._cur_state = state

        return transition, Context(policy_info=info)

    def run(self, max_steps=-1):
        """Python generator. Play agent in environment.

        Args:
            max_steps (int): Play until env is done or max_steps is reached.
        If -1, then play until env is done. [Default: -1]

        Returns:
            transition (environments.Transition): Transition packed in namedtuple: 
        state, action, reward, next_state, is_terminal.
            context (agents.Context): Includes user defined object,
        returned from policy.
        """

        stop = False
        step = 0

        while not stop and (max_steps == -1 or step < max_steps):
            transition, info = self.step()
            stop = transition.is_terminal

            yield transition, info
            step += 1