import logging as log
import numpy as np

from collections import namedtuple
from multiprocessing import Pool
from tqdm import tqdm

from .agents import Vision
from .callbacks import CallbackList
from .utils import History, unpack

Transition = namedtuple(
    "Transition", ["player", "state", "action", "reward", "next_player", "next_state", "is_terminal"])


class Worker(object):
    """Loop worker."""

    def __init__(self, env_factory, mind_factory, loop_kwargs):
        self.env_factory = env_factory
        self.mind_factory = mind_factory
        self.loop_kwargs = loop_kwargs

        self.env = None

    def __call__(self, job):
        if self.env is None:
            self.env = self.env_factory()

        return loop(self.env, self.mind_factory(job), **self.loop_kwargs)


def ply(env, mind, policy='deterministic', vision=None, step=0, train_mode=True,
        debug_mode=False, callbacks=None, **kwargs):
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
            See `Callback` class docstrings. (Default: None)
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

    # Create default vision if one wasn't passed
    vision = vision or Vision()
    
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


def loop(env, minds, vision=None, n_episodes=1, max_steps=-1, policy='deterministic', name="",
         alternate_players=False, debug_mode=False, render_mode=False, train_mode=True,
         verbose=2, callbacks=None, **kwargs):
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
            See `Callback` class docstrings. (Default: None)
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
        log.critical("Safely terminating loop...")
        callbacks_list.on_loop_end(True)
        raise  # Pass exception upper

    # Finish loop as planned
    callbacks_list.on_loop_end(False)
    return history.history


def pool(env_factory, mind_factory, jobs, processes=None, **kwargs):
    """Runs `processes` number of workers which executes jobs (minds instances) in own environments.

    Args:
        env_factory (callable): Function that takes no arguments and return 'Environment'.
            Called once per worker and shared between jobs.
        mind_factory (callable): Function that takes one argument, job element and returns 'Mind'.
            Called once per each job ('hrl.loop(...)' execution on environment with mind).
        jobs (iterable): Jobs (picklable objects) run by workers. Can be e.g. 'range(...)'.
        processes (int or None): Number of workers. If None then taken from 'os.cpu_count()'.
            (Default: None)
        **kwargs: Arguments passed to each 'hrl.loop(...)' execution. Copied to each worker.

    Returns:
        list: History dictionaries from 'humblerl.loop(...)' of each job.
    """

    worker = Worker(env_factory, mind_factory, kwargs)
    with Pool(processes=processes) as pool:
        return pool.map(worker, jobs)
