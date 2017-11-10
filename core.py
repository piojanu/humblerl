import numpy as np

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from third_party.progress.bar import Bar

Transition = namedtuple(
    "Transition", ["player", "state", "action", "reward", "next_player", "next_state", "is_terminal"])


class Callback(metaclass=ABCMeta):
    """Callbacks can be used to listen to events during :func:`loop`."""

    @abstractmethod
    def on_reset(self, train_mode):
        """Event after environment reset.

        Args:
            train_mode (bool): Informs whether environment is in training or evaluation mode.
        """

        pass

    @abstractmethod
    def on_step(self, transition, info):
        """Event after action taken in environment.

        Args:
            transition (Transition): Describes transition that took place.
            info (object): Meta information obtained from Mind.

        Note:
            Transition and info are returned from `ply` function (look to docstring for more info).
        """

        pass


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    @abstractmethod
    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. (Default: True)

        Returns:
            np.Array: The initial state. 
            int: Current player, first is 0.

        Note:
            In child class you MUST set `self._curr_state` to returned initial state.
        """

        # TODO (pj): Is there a better way to set `self._curr_state` then telling user
        #            to do this manually?
        pass

    @abstractmethod
    def step(self, action):
        """Perform action in environment.

        Args:
            action (list of floats): Action to perform. In discrete action space it's single
        element list with action number. In continuous case, it's action vector.

        Returns:
            np.Array: New state.
            int: Current player, first is 0.
            float: Next reward.
            bool: Flag indicating if episode has ended.

        Note:
            In child class you MUST set `self._curr_state` to returned new state.
        """

        # TODO (pj): Is there a better way to set `self._curr_state` then telling user
        #            to do this manually?
        pass

    @property
    def current_state(self):
        """Access state.

        Returns:
            np.array: Current environment state.
        """

        return self._curr_state

    @property
    @abstractmethod
    def valid_actions(self):
        """Access valid actions.

        Returns:
            np.array: Array with indexes of currently available actions.
        """

        pass


class Mind(metaclass=ABCMeta):
    """Artificial mind of RL agent."""

    @abstractmethod
    def plan(self, state, player):
        """Do forward pass through agent model, inference/planning on state.

        Args:
            state (numpy.Array): State of game to inference on.
            player (int): Current player index.

        Returns:
            numpy.Array: Actions scores (e.g. action unnormalized log probabilities/Q-values/etc.).
            object: Meta information which can be accessed later with transition.
        """

        pass


class Vision(object):
    """Vision system entity in Reinforcement Learning task.

       It is responsible for data preprocessing.
    """

    def __init__(self, state_processor_fn=None, reward_processor_fn=None):
        """Initialize vision processors.

        Args:
            state_processor_fn (function): Function for state processing. It should
        take raw environment state as an input and return processed state.
        (Default: None which will result in passing raw state)
            reward_processor_fn (function): Function for reward processing. It should
        take raw environment reward as an input and return processed reward.
        (Default: None which will result in passing raw reward)
        """

        self._process_state = \
            state_processor_fn if state_processor_fn is not None else lambda x: x
        self._process_reward = \
            reward_processor_fn if reward_processor_fn is not None else lambda x: x

    def __call__(self, state, reward=0.):
        return self._process_state(state), self._process_reward(reward)


def ply(env, mind, player=0, policy='deterministic', vision=Vision(), step=0, **kwargs):
    """Conduct single ply (turn taken by one of the players).

    Args:
        env (Environment): Environment to take actions in.
        mind (Mind): Mind to use while deciding on action to take in the env.
        player (int): Player index which ply this is. (Default: 0)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        step (int): Current step number in this episode, used by some policies. (Default: 0)
        **kwargs: Other keyword arguments may be needed depending on chosen policy.

    Return:
        Transition: Describes transition that took place. It contains:
          * 'player'       : player index which ply this is (zero is first),
          * 'state'        : state from which transition has started (it's preprocessed with Vision),
          * 'action'       : action taken (chosen by policy),
          * 'reward'       : reward obtained (it's preprocessed with Vision),
          * 'next_player'  : next player index,
          * 'next_state'   : next state observed after transition (it's preprocessed with Vision),
          * 'is_terminal'  : flag indication if this is terminal transition (episode end).
        object: Meta information obtained from Mind.

    Note:
        Possible :attr:`policy` values are:
          * 'deterministic': default, pass extra kwarg :attr:`warmup` to use stochastic policy
                             during first steps until step < :attr:`warmup`.
                             Stochastic annealing also apply.
          * 'stochastic'   : pass extra kwarg 'temperature', otherwise it's set to 1.
                             You can also anneal temperature using :attr:`decay`:
                             temp * (1. / (1. + decay * step)).
          * 'egreedy'      : pass extra kwarg 'epsilon', otherwise it's set to 0.5.
                             You can also anneal epsilon using :attr:`decay`:
                             epsilon * (1. / (1. + decay * step)).
          * 'identity'     : forward whatever come from Mind.
    """

    # Get and preprocess current state
    curr_state, _ = vision(env.current_state)

    # Infer what to do
    logits, info = mind.plan(curr_state, player)

    # Get valid actions and logits of those actions
    valid_actions = env.valid_actions
    valid_logits = np.take(logits, valid_actions)

    # Define policies
    def deterministic():
        return valid_actions[np.argmax(valid_logits)]

    def stochastic():
        temp = kwargs.get('temperature', 1.)
        decay = kwargs.get('decay', 0.)

        # Decay temperature
        if decay > 0:
            temp *= 1. / (1. + decay * step)

        # Softmax with temperature
        exps = np.exp((valid_logits - np.max(valid_logits)) / temp)
        probs = exps / np.sum(exps)

        # Sample from created distribution
        return np.random.choice(valid_actions, p=probs)

    def egreedy():
        eps = kwargs.get('epsilon', 0.5)
        decay = kwargs.get('decay', 0.)

        # Decay epsilon
        if decay > 0:
            epsilon *= 1. / (1. + decay * step)

        # With probability of epsilon...
        if np.random.rand(1) < eps:
            # ...sample random action, otherwise
            return np.random.choice(valid_actions)
        else:
            # ...choose action greedily
            return valid_actions[np.argmax(valid_logits)]

    # Get action
    if policy == 'deterministic':
        warmup = kwargs.get('warmup', 0)
        if step < warmup:
            action = stochastic()
        else:
            action = deterministic()
    elif policy == 'stochastic':
        action = stochastic()
    elif policy == 'egreedy':
        action = egreedy()
    elif policy == 'identity':
        action = logits
    else:
        raise ValueError("Undefined policy")

    # Take chosen action
    raw_next_state, next_player, raw_reward, done = env.step(action)

    # Preprocess data and save in transition
    next_state, reward = vision(raw_next_state, raw_reward)
    transition = Transition(player, curr_state, action, reward, next_player, next_state, done)

    return transition, info


def loop(env, minds, n_episodes=1, max_steps=-1, alternate_minds=False, policy='deterministic', train_mode=True, vision=Vision(), name="Loop", callbacks=[], **kwargs):
    """Conduct series of plies (turns taken by each player in order).

    Args:
        env (Environment): Environment to take actions in.
        minds (Mind or list of Mind objects): Minds to use while deciding on action to take in the env.
    If more then one, then each will be used one by one starting form index 0.
        alternate_minds (bool): If minds order should be alternated or left unchanged in each
    episode. (Default: False)
        n_episodes (int): Number of episodes to play. (Default: 1)
        max_steps (int): Maximum number of steps in episode. No limit when -1. (Default: -1)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        train_mode (bool): Informs environment whether it's in training or evaluation mode.
    E.g. in train mode graphics could not be rendered. (Default: True)
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        name (string): Name shown in progress bar. (Default: "Loop")
        callbacks (list of Callback objects): Objects that can listen to events during play.
    (Default: [])
        **kwargs: Other keyword arguments may be needed depending on chosen policy.

    Note:
        Possible `policy` values are:
          * 'deterministic': default,
          * 'stochastic'   : pass extra kwarg 'temperature' otherwise it's set to 1.,
          * 'egreedy'      : pass extra kwarg 'epsilon' otherwise it's set to 0.5,
          * 'identity'     : forward whatever come from Mind.
    """

    # Play given number of episodes
    bar = Bar(name, suffix='%(index)d/%(max)d - %(avg).3fs/episode, ETA: %(eta)ds')
    for _ in bar.iter(range(n_episodes)):
        step = 0
        _, player = env.reset(train_mode)

        # Alternate minds in list
        if alternate_minds:
            minds.append(minds.pop(0))

        # Callback reset
        for callback in callbacks:
            callback.on_reset(train_mode)

        # Play until episode ends or max_steps limit reached
        while max_steps == -1 or step <= max_steps:
            # Determine player index and mind
            if isinstance(minds, (list, tuple)):
                mind = minds[player]
            else:
                mind = minds

            # Conduct ply and update next player
            transition, info = ply(env, mind, player, policy, vision, step, **kwargs)
            player = transition.next_player

            # Callback step and increment step counter
            for callback in callbacks:
                callback.on_step(transition, info)
            step += 1

            # Finish if this transition was terminal
            if transition.is_terminal:
                break
