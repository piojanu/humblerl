import logging as log
import numpy as np
import sys

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from tqdm import tqdm

Transition = namedtuple(
    "Transition", ["player", "state", "action", "reward", "next_player", "next_state", "is_terminal"])


class Callback(metaclass=ABCMeta):
    """Callbacks can be used to listen to events during RL loop execution."""

    def on_loop_start(self):
        """Event when loop starts.

        Note:
            You can assume, that this event occurs before any other event in current loop.
        """

        pass

    def on_loop_end(self, is_aborted):
        """Event when loop finish.

        Args:
            is_aborted (bool): Flag indication if loop has finished as planed or was terminated.

        Note:
            You can assume, that this event occurs after specified episodes number or when
            loop is terminated manually (e.g. by Ctrl+C).
        """

        pass

    def on_episode_start(self, episode, train_mode):
        """Event when episode starts.

        Args:
            episode (int): Episode number.
            train_mode (bool): Informs whether episode is in training or evaluation mode.

        Note:
            You can assume, that this event occurs always before any action is taken in episode.
        """

        pass

    def on_episode_end(self, episode, train_mode):
        """Event after environment was reset.
        
        Args:
            episode (int): Episode number.
            train_mode (bool): Informs whether episode is in training or evaluation mode.

        Note:
            You can assume, that this event occurs after step to terminal state.
        """

        pass

    def on_action_planned(self, step, logits, info):
        """Event after Mind was evaluated.

        Args:
            step (int): Step number.
            logits (np.array): Actions scores (e.g. unnormalized log probabilities/Q-values/etc.)
                raw values returned from 'Mind.plan(...)'.
            info (object): Mind's extra information, may be None.

        Note:
            You can assume, that this event occurs always before step finish.
        """

        pass

    def on_step_taken(self, step, transition, info):
        """Event after action was taken in environment.

        Args:
            step (int): Step number.
            transition (Transition): Describes transition that took place.
            info (object): Environment diagnostic information if available otherwise None.

        Note:
            Transition is returned from `ply` function (look to docstring for more info).
            Also, you can assume, that this event occurs always after action was planned.
        """

        pass

    @property
    def metrics(self):
        """Returns execution metrics.

        Returns:
            dict: Dictionary with logs names and values.
        
        Note:
            Those values are fetched by 'humblerl.loop(...)' at the end of each episode (after
            'on_episode_end is' called) and then returned from 'humblerl.loop(...)' as evaluation
            history. Those also are logged by 'humblerl.loop(...)' depending on its verbosity.
        """

        return {}


class CallbackList(object):
    """Simplifies calling all callbacks from list."""

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_loop_start(self):
        for callback in self.callbacks:
            callback.on_loop_start()

    def on_loop_end(self, is_aborted):
        for callback in self.callbacks:
            callback.on_loop_end(is_aborted)

    def on_episode_start(self, episode, train_mode):
        for callback in self.callbacks:
            callback.on_episode_start(episode, train_mode)

    def on_episode_end(self, episode, train_mode):
        for callback in self.callbacks:
            callback.on_episode_end(episode, train_mode)

    def on_action_planned(self, step, logits, info):
        for callback in self.callbacks:
            callback.on_action_planned(step, logits, info)

    def on_step_taken(self, step, transition, info):
        for callback in self.callbacks:
            callback.on_step_taken(step, transition, info)

    @property
    def metrics(self):
        metrics = {}
        for callback in self.callbacks:
            metrics.update(callback.metrics)
        return metrics

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
    def state_space(self):
        """Access environment state space.

        Returns:
            np.array: If desecrate state space, then it's one item describing state space size.
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
            np.array: If desecrate action space, then it's a 1D array with available action values.
                If continuous, then this is 2D array, where first dimension has action vector size
                and second dimension of size 2 keeps respectively [min, max] (inclusive range)
                values which given action vector element can take.

        Note:
            In child class just set `self._valid_actions`. If valid actions depend on current
            state, just override this property.
        """

        return self._valid_actions


class Mind(metaclass=ABCMeta):
    """Artificial mind of RL agent."""

    @abstractmethod
    def plan(self, state, player, train_mode, debug_mode):
        """Do forward pass through agent model, inference/planning on state.

        Args:
            state (np.array): State of environment to inference on.
            player (int): Current player index.
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
            np.array: If desecrate state space, then it's one item describing state space size.
                If continuous, then this is (M + 1)-D array, where first M dimensions are
                state dimensions and last dimension of size 2 keeps respectively [min, max]
                (inclusive range) values which given state feature can take.
        """

        pass


class Vision(object):
    """Vision system entity in Reinforcement Learning task.

       It is responsible for e.g. data preprocessing, feature extraction etc.
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


def ply(env, mind, policy='deterministic', vision=Vision(), step=0, train_mode=True,
        debug_mode=False, callbacks=[], **kwargs):
    """Conduct single ply (turn taken by one of the players).

    Args:
        env (Environment): Environment to take actions in.
        mind (Mind): Mind to use while deciding on action to take in the env.
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
          * 'proportional' : Same as stochastic, but you normalize logits not exponential.
          * 'egreedy'      : pass extra kwarg 'epsilon', otherwise it's set to 0.5.
                             You can also anneal epsilon using :attr:`decay`:
                             epsilon * (1. / (1. + decay * step)).
          * 'identity'     : forward whatever come from Mind.
    """

    # Create callbacks list
    callbacks_list = CallbackList(callbacks)

    # Get current player and current state (preprocess it)
    curr_player = env.current_player
    curr_state, _ = vision(env.current_state)

    # Infer what to do
    logits, mind_info = unpack(mind.plan(curr_state, curr_player, train_mode, debug_mode))
    callbacks_list.on_action_planned(step, logits, mind_info)

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
            # ...sample random action, otherwise...
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
    raw_next_state, next_player, raw_reward, done, env_info = env.step(action)

    # Preprocess data and save in transition
    next_state, reward = vision(raw_next_state, raw_reward)
    transition = Transition(curr_player, curr_state, action, reward, next_player, next_state, done)
    callbacks_list.on_step_taken(step, transition, env_info)

    return transition


def loop(env, minds, vision=Vision(), n_episodes=1, max_steps=-1, policy='deterministic', name="",
         alternate_players=False, debug_mode=False, render_mode=False, train_mode=True,
         verbose=2, callbacks=[], **kwargs):
    """Conduct series of plies (turns taken by each player in order).

    Args:
        env (Environment): Environment to take actions in.
        minds (Mind or list of Mind objects): Minds to use while deciding on action to take in the
            environment. If list, then number of minds must be equal to number of players. Then each
            mind is chosen according to current player index.
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        n_episodes (int): Number of episodes to play. (Default: 1)
        max_steps (int): Maximum number of steps in episode. No limit when -1. (Default: -1)
        policy (string): Describes the way of choosing action from mind predictions
            (see Note section in docstring of `ply` function).
        name (string): Name shown in progress bar. (Default: "")
        alternate_players (bool): If players order should be alternated or left unchanged in each
            episode. It controls if starting player should change after episode. (Default: False)
        debug_mode (bool): Informs Mind whether it's in debug mode or not. (Default: False)
        render_mode (bool): If environment should be rendered. (Default: False)
        train_mode (bool): Informs env and Mind whether they're in training or evaluation mode.
            (Default: True)
        verbose (int): Specify how much information to log:
            0: nothing, 1: progress bar with last episode metrics, 2: each episode metrics.
            (Default: 2)
        callbacks (list of Callback objects): Objects that can listen to events during play.
            See `Callback` class docstrings. (Default: [])
        **kwargs: Other keyword arguments may be needed depending on e.g. chosen policy.

    Returns:
        dict of lists: Evaluation history. Those are callbacks metrics gathered through course of
            loop execution. Keys are metrics names and metrics values for each episode are in lists.
    """

    # Create callbacks list and "officially start loop"
    callbacks_list = CallbackList(callbacks)
    callbacks_list.on_loop_start()

    # Create history object to store metrics after each episode
    history = History()

    try:
        # Play given number of episodes
        first_player = 0
        pbar = tqdm(range(n_episodes), ascii=True, desc=name,
                    disable=True if verbose == 0 else False)
        for itr in pbar:
            step = 0
            _, player = env.reset(train_mode, first_player=first_player)
            callbacks_list.on_episode_start(itr, train_mode)

            # Play until episode ends or max_steps limit reached
            while max_steps == -1 or step <= max_steps:
                # Render environment if requested
                if render_mode:
                    env.render()

                # Determine player index and mind
                if isinstance(minds, (list, tuple)):
                    mind = minds[player]
                else:
                    mind = minds

                # Conduct ply and update next player
                transition = ply(
                    env, mind, policy, vision, step, train_mode, debug_mode, callbacks, **kwargs)
                player = transition.next_player

                # Increment step counter
                step += 1

                # Finish if this transition was terminal
                if transition.is_terminal:
                    callbacks_list.on_episode_end(itr, train_mode)
                    metrics = callbacks_list.metrics
                    history.update(metrics)

                    if verbose == 1:
                        # Update bar suffix
                        pbar.set_postfix(metrics)
                    elif verbose >= 2:
                        # Write episode metrics
                        pbar.write("Episode {:2}/{}: ".format(itr + 1, n_episodes) + ", ".join(
                            ["{}: {:.4g}".format(k, float(v)) for k, v in metrics.items()]))

                    # Finish episode
                    break

            # Change first player
            if isinstance(minds, (list, tuple)) and alternate_players:
                first_player = (first_player + 1) % len(minds)
    except KeyboardInterrupt:
        # Finish loop when aborted
        log.critical("KeyboardInterrupt, safely terminate loop and exit")
        callbacks_list.on_loop_end(True)
        sys.exit()

    # Finish loop as planned
    callbacks_list.on_loop_end(False)
    return history.history

# This import has to be in here because of circular import between 'core' and 'utils'
from .utils import History, unpack
