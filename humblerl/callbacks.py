import csv
import numpy as np
import os

from abc import ABCMeta


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
        self.callbacks = callbacks or []

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


class CSVSaverWrapper(CallbackList):
    """Saves to .csv file whatever wrapped callback logs.

    Args:
        callback (Callback): Source callback to save logs from.
        path (string): Where to save logs.
        only_last (bool): If only save last log in the loop. Useful when source
            callback aggregates logs. (Default: False)
    """

    def __init__(self, callback, path, only_last=False):
        super(CSVSaverWrapper, self).__init__([callback])

        self.path = path
        self.only_last = only_last
        self.history = []

    def on_episode_end(self, episode, train_mode):
        self.unwrapped.on_episode_end(episode, train_mode)
        self.history.append(self.unwrapped.metrics)

    def on_loop_end(self, is_aborted):
        if len(self.history) > 0:
            self._store()
        self.unwrapped.on_loop_end(is_aborted)

    @property
    def unwrapped(self):
        return self.callbacks[0]

    def _store(self):
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            with open(self.path, "w") as f:
                logs_file = csv.DictWriter(f, self.history[0].keys())
                logs_file.writeheader()

        with open(self.path, "a") as f:
            logs_file = csv.DictWriter(f, self.history[0].keys())
            if self.only_last:
                logs_file.writerow(self.history[-1])
            else:
                logs_file.writerows(self.history)
            self.history = []


class BasicStats(Callback):
    """Gather basic episode statistics like:
      * number of steps,
      * return,
      * max reward,
      * min reward.
    """

    def on_episode_start(self, episode, train_mode):
        self._reset()

    def on_step_taken(self, step, transition, info):
        self.steps += 1
        self.rewards.append(transition.reward)

    @property
    def metrics(self):
        logs = {}
        logs["# steps"] = self.steps
        logs["return"] = np.sum(self.rewards)
        logs["max reward"] = np.max(self.rewards)
        logs["min reward"] = np.min(self.rewards)

        return logs

    def _reset(self):
        self.steps = 0
        self.rewards = []
