class Vision(object):
    """Vision system entity in Reinforcement Learning task."""

    def __init__(self, state_processor=None, reward_processor=None):
        """Initialize vision processors.

        Args:
            state_processor (function): Function for state processing. It should
        take raw environment state as an input and return processed state.
        Default: None which will result in passing raw state.
            reward_processor (function): Function for reward processing. It should
        take raw environment reward as an input and return processed reward.
        Default: None which will result in passing raw reward.
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
        before passing it to the agent. Default: humblerl.agents.Vision().
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
        mode. E.g. in train mode graphics could not be rendered. (default: True)

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
        action (list of floats) to take in the environment. In discrete action
        space it's single element list with action number.
        If None, previous policy will be used (it's called current policy).
        Default: None.

        Returns:
            list of floats: Action taken. In discrete action space it's single
        element list with action number.
            np.array: Next state.
            float: Next reward.
            bool: Flag indicating if episode has ended.
        """

        # Assign policy
        if policy is not None:
            self._cur_policy = policy

        # Checks if everything needed to take step is present
        if self._cur_state is None:
            raise ValueError("You need to reset agent first!")

        if self._cur_policy is None:
            raise ValueError("You need to provide agent policy!")

        # Take a step
        action = self._cur_policy(self._cur_state)
        raw_state, raw_reward, done = self._env.step(action=action)

        # Process raw state and reward
        state, reward = self._vision(raw_state, raw_reward)
        self._cur_state = state

        return action, state, reward, done
