import logging as log
import numpy as np
import sys

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from tqdm import tqdm

Transition = namedtuple(
    "Transition", ["player", "state", "action", "reward", "next_player", "next_state", "is_terminal"])


class Callback(metaclass=ABCMeta):
    """Callbacks can be used to listen to events during :func:`loop`."""

    def on_loop_start(self):
        """Event when loop starts.

        Note:
            You can assume, that this event occurs before any other event in current loop.
        """

        pass

    def on_episode_start(self, train_mode):
        """Event when episode starts.

        Args:
            train_mode (bool): Informs whether episode is in training or evaluation mode.

        Note:
            You can assume, that this event occurs always before any action is taken in episode.
        """

        pass

    def on_action_planned(self, logits, metrics):
        """Event after Mind was evaluated.

        Args:
            logits (numpy.Array): Actions scores (e.g. unnormalized log probabilities/Q-values/etc.)
        possibly raw Artificial Neural Net output i.e. logits.
            metrics (dict): Planning metrics.

        Note:
            You can assume, that this event occurs always before step finish.
        """

        pass

    def on_step_taken(self, transition):
        """Event after action was taken in environment.

        Args:
            transition (Transition): Describes transition that took place.

        Note:
            Transition and info are returned from `ply` function (look to docstring for more info).
            Also, you can assume, that this event occurs always after action planned.
        """

        pass

    def on_episode_end(self):
        """Event after environment was reset.

        Returns:
            dict: Dictionary with logs names and values. Those may be visible in CMD progress bar
        and saved to log file if specified.

        Note:
            You can assume, that this event occurs after step to terminal state.
        """

        return {}

    def on_loop_finish(self, is_aborted):
        """Event after  was reset.

        Args:
            is_aborted (bool): Flag indication if loop has finished as planed or was terminated.

        Note:
            You can assume, that this event occurs after specified episodes number or when
        loop is terminated manually (e.g. by Ctrl+C).
        """

        pass


class CallbackList(object):
    """Simplifies calling all callbacks from list."""

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_loop_start(self):
        for callback in self.callbacks:
            callback.on_loop_start()

    def on_episode_start(self, train_mode):
        for callback in self.callbacks:
            callback.on_episode_start(train_mode)

    def on_action_planned(self, logits, metrics):
        for callback in self.callbacks:
            callback.on_action_planned(logits, metrics)

    def on_step_taken(self, transition):
        for callback in self.callbacks:
            callback.on_step_taken(transition)

    def on_episode_end(self):
        logs = {}
        for callback in self.callbacks:
            logs.update(callback.on_episode_end())
        return logs

    def on_loop_finish(self, is_aborted):
        for callback in self.callbacks:
            callback.on_loop_finish(is_aborted)


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    @abstractmethod
    def reset(self, train_mode=True, first_player=0):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. (Default: True)
            first_player (int): Index of player who starts game. (Default: 0)

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
            float: Reward.
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
    def plan(self, state, player, train_mode, debug_mode):
        """Do forward pass through agent model, inference/planning on state.

        Args:
            state (numpy.Array): State of game to inference on.
            player (int): Current player index.
            train_mode (bool): Informs planner whether it's in training or evaluation mode.
        E.g. in evaluation it can optimise graph, disable exploration etc.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            numpy.Array: Actions scores (e.g. unnormalized log probabilities/Q-values/etc.)
        possibly raw Artificial Neural Net output i.e. logits.
            dict: Planning metrics, content possibly depended on :param:`debug_mode` value.
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


def ply(env, mind, player=0, policy='deterministic', vision=Vision(), step=0, train_mode=True,
        debug_mode=False, callbacks=[], **kwargs):
    """Conduct single ply (turn taken by one of the players).

    Args:
        env (Environment): Environment to take actions in.
        mind (Mind): Mind to use while deciding on action to take in the env.
        player (int): Player index which ply this is. (Default: 0)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        step (int): Current step number in this episode, used by some policies. (Default: 0)
        train_mode (bool): Informs env and planner whether it's in training or evaluation mode.
    (Default: True)
        debug_mode (bool): Informs planner whether it's in debug mode or not. (Default: False)
        callbacks (list of Callback objects): Objects that can listen to events during play.
    (Default: [])
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

    Note:
        Possible :attr:`policy` values are:
          * 'deterministic': default, also used after step >= :attr:`warmup` in stochastic and
                             proportional policies. If this is a case, then temperature and
                             annealing also apply.
          * 'stochastic'   : pass extra kwarg 'temperature', otherwise it's set to 1.
                             You can also anneal temperature using :attr:`decay`:
                             temp * (1. / (1. + decay * step)).
          * 'egreedy'      : pass extra kwarg 'epsilon', otherwise it's set to 0.5.
                             You can also anneal epsilon using :attr:`decay`:
                             epsilon * (1. / (1. + decay * step)).
          * 'identity'     : forward whatever come from Mind.
    """

    # Create callbacks list
    callbacks_list = CallbackList(callbacks)

    # Get and preprocess current state
    curr_state, _ = vision(env.current_state)

    # Infer what to do
    logits, metrics = mind.plan(curr_state, player, train_mode, debug_mode)
    callbacks_list.on_action_planned(logits, metrics)

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

    def proportional():
        temp = kwargs.get('temperature', 1.)
        decay = kwargs.get('decay', 0.)

        # Ensure that all values starts from 0
        np.maximum(valid_logits, 0, valid_logits)

        # Decay temperature
        if decay > 0:
            temp *= 1. / (1. + decay * step)

        # Normalized with temperature
        exps = np.power(valid_logits, 1. / temp)
        probs = exps / np.sum(exps)

        # Sample from created distribution
        return np.random.choice(valid_actions, p=probs)

    def egreedy():
        eps = kwargs.get('epsilon', 0.5)
        decay = kwargs.get('decay', 0.)

        # Decay epsilon
        if decay > 0:
            eps *= 1. / (1. + decay * step)

        # With probability of epsilon...
        if np.random.rand(1) < eps:
            # ...sample random action, otherwise
            return np.random.choice(valid_actions)
        else:
            # ...choose action greedily
            return valid_actions[np.argmax(valid_logits)]

    # Get action
    if policy == 'deterministic':
        action = deterministic()
    elif policy == 'stochastic':
        warmup = kwargs.get('warmup', 0)
        if step < warmup:
            action = stochastic()
        else:
            action = deterministic()
    elif policy == 'proportional':
        warmup = kwargs.get('warmup', 0)
        if step < warmup:
            action = proportional()
        else:
            action = deterministic()
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
    callbacks_list.on_step_taken(transition)

    return transition


def loop(env, minds, n_episodes=1, max_steps=-1, alternate_players=False, policy='deterministic',
         vision=Vision(), name="Loop", train_mode=True, debug_mode=False, verbose=1, callbacks=[],
         **kwargs):
    """Conduct series of plies (turns taken by each player in order).

    Args:
        env (Environment): Environment to take actions in.
        minds (Mind or list of Mind objects): Minds to use while deciding on action to take in the env.
    If more then one, then each will be used one by one starting form index 0.
        alternate_players (bool): If players order should be alternated or left unchanged in each
    episode. (Default: False)
        n_episodes (int): Number of episodes to play. (Default: 1)
        max_steps (int): Maximum number of steps in episode. No limit when -1. (Default: -1)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        name (string): Name shown in progress bar. (Default: "Loop")
        train_mode (bool): Informs env and planner whether it's in training or evaluation mode.
    (Default: True)
        debug_mode (bool): Informs planner whether it's in debug mode or not. (Default: False)
        verbose (int): Specify how much information to log (0: nothing, 1: progress bar, 2: logs).
    (Default: 1)
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

    # Create callbacks list and "officially start loop"
    callbacks_list = CallbackList(callbacks)
    callbacks_list.on_loop_start()

    try:
        # Play given number of episodes
        first_player = 0
        pbar = tqdm(range(n_episodes), ascii=True, desc=name,
                    disable=True if verbose == 0 else False)
        for iter in pbar:
            step = 0
            _, player = env.reset(train_mode, first_player=first_player)
            callbacks_list.on_episode_start(train_mode)

            # Play until episode ends or max_steps limit reached
            while max_steps == -1 or step <= max_steps:
                # Determine player index and mind
                if isinstance(minds, (list, tuple)):
                    mind = minds[player]
                else:
                    mind = minds

                # Conduct ply and update next player
                transition = ply(
                    env, mind, player, policy, vision, step, train_mode, debug_mode, callbacks, **kwargs)
                player = transition.next_player

                # Increment step counter
                step += 1

                # Finish if this transition was terminal
                if transition.is_terminal:
                    logs = callbacks_list.on_episode_end()
                    if verbose >= 2:
                        # Update bar suffix
                        pbar.write("Iter. {:3}".format(iter) + ": [ " + ", ".join(
                            ["{}: {:.4g}".format(k, float(v)) for k, v in logs.items()]) + " ]")

                    # Finish episode
                    break

            # Change first player
            if isinstance(minds, (list, tuple)) and alternate_players:
                first_player = (first_player + 1) % len(minds)
    except KeyboardInterrupt:
        # Finish loop when aborted
        log.critical("KeyboardInterrupt, safely terminate loop and exit")
        callbacks_list.on_loop_finish(True)
        sys.exit()

    # Finish loop as planned
    callbacks_list.on_loop_finish(False)
