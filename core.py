import logging as log
import numpy as np

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from multiprocessing import Pool
from tqdm import tqdm

from .agents import Vision
from .callbacks import CallbackList
from .utils import History, unpack

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "is_terminal"])


class Worker(metaclass=ABCMeta):
    """Worker to prepare environment, vision and callbacks for loop.
    Also provides minds factory from jobs."""

    @abstractmethod
    def initialize(self):
        """Called once per worker, should create environment and put it in 'self._env' member.
        Can also provide vision and callbacks for loop. See 'vision' and 'callbacks' properties
        which you need to override."""

        pass

    @abstractmethod
    def mind_factory(self, job):
        """Called once per each job ('hrl.loop(...)' execution).

        Args:
            job (object): Provided to 'hrl.pool(...)' jobs iterable element.

        Returns:
            Mind: mind to use during job evaluation.
        """

        pass

    @property
    def environment(self):
        """This is just a getter.

        Returns:
            Environment: environment to evaluate minds in.

        Note:
            Environment should be constructed in 'Worker.initialize()' method.
        """

        # 'self._env' should be set in 'Worker.initialize()'.
        return self._env

    @property
    def callbacks(self):
        """This is just a getter.

        Returns:
            list: list of callbacks.

        Note:
            Callbacks should be constructed in 'Worker.initialize()' method.
        """

        # By default no callbacks.
        return []

    @property
    def vision(self):
        """This is just a getter.

        Returns:
            Vision: humbler vision used to preprocess states and rewards during evaluation.

        Note:
            Vision should be constructed in 'Worker.initialize()' method.
        """

        # By default no preprocessing.
        return Vision()


def initializer(worker, loop_kwargs):
    global w, kwargs
    kwargs = loop_kwargs
    w = worker
    w.initialize()


def evaluate(job):
    global w, kwargs

    return loop(w.environment, w.mind_factory(job),
                vision=w.vision, callbacks=w.callbacks,
                **kwargs)


def ply(env, mind, policy='deterministic', vision=None, step=0, train_mode=True,
        debug_mode=False, callbacks=None, **kwargs):
    """Conduct single ply (e.g. step in env or turn taken by one of the players, ...).

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
          * 'state'        : state from which transition has started (it's preprocessed with Vision),
          * 'action'       : action taken (chosen by policy),
          * 'reward'       : reward obtained (it's preprocessed with Vision),
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

    # Get current state (preprocess it)
    curr_state, _ = vision(env.current_state)

    # Infer what to do
    logits, mind_info = unpack(mind.plan(curr_state, train_mode, debug_mode))
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
    raw_next_state, raw_reward, done, env_info = env.step(action)

    # Preprocess data and save in transition
    next_state, reward = vision(raw_next_state, raw_reward)
    transition = Transition(curr_state, action, reward, next_state, done)
    callbacks_list.on_step_taken(step, transition, env_info)

    return transition


def loop(env, mind, vision=None, n_episodes=1, max_steps=-1, policy='deterministic', name="",
         debug_mode=False, render_mode=False, train_mode=True, verbose=2, callbacks=None, **kwargs):
    """Conduct series of plies.

    Args:
        env (Environment): Environment to take actions in.
        mind (Mind): Mind to use while deciding on action to take in the environment.
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        n_episodes (int): Number of episodes to play. (Default: 1)
        max_steps (int): Maximum number of steps in episode. No limit when -1. (Default: -1)
        policy (string): Describes the way of choosing action from mind predictions
            (see Note section in docstring of `ply` function).
        name (string): Name shown in progress bar. (Default: "")
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

    Note: When an exception is handled during loop execution, exception is thrown out of the function.

    """

    # Create callbacks list and "officially start loop"
    callbacks_list = CallbackList(callbacks)
    callbacks_list.on_loop_start()

    # Create history object to store metrics after each episode
    history = History()

    try:
        # Play given number of episodes
        pbar = tqdm(range(n_episodes), ascii=True, desc=name,
                    disable=True if verbose == 0 else False)
        for itr in pbar:
            step = 0
            env.reset(train_mode)
            callbacks_list.on_episode_start(itr, train_mode)

            # Play until episode ends or max_steps limit reached
            while max_steps == -1 or step <= max_steps:
                # Render environment if requested
                if render_mode:
                    env.render()

                # Play
                transition = ply(
                    env, mind, policy, vision, step, train_mode, debug_mode, callbacks, **kwargs)

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

    except Exception as err:
        # Finish loop when aborted
        log.critical("{}\nSafely terminating loop...".format(err))
        callbacks_list.on_loop_end(True)
        raise err

    # Finish loop as planned
    callbacks_list.on_loop_end(False)
    return history.history


def pool(worker, jobs, processes=None, **kwargs):
    """Runs `processes` number of workers which executes jobs (minds instances) in own environments.

    Args:
        worker (Worker): Provides env, vision and callbacks list to each worker. It also implement
            minds factory. It needs to be picklable.
        jobs (iterable): Jobs (picklable objects) run by workers. Can be e.g. 'range(...)'.
        processes (int or None): Number of workers. If None then taken from 'os.cpu_count()'.
            (Default: None)
        **kwargs: Arguments passed to each 'hrl.loop(...)' execution. Copied to each worker.
            You shouldn't provide 'vision' and 'callbacks' arguments. Use Factory object for this.

    Returns:
        list: History dictionaries from 'humblerl.loop(...)' of each job.
    """

    assert 'vision' not in kwargs and 'callbacks' not in kwargs, \
        "You shouldn't provide 'vision' and 'callbacks' arguments."

    with Pool(processes=processes, initializer=initializer, initargs=(worker, kwargs)) as pool:
        return pool.map(evaluate, jobs)
