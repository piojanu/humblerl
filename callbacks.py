import csv
import os
import numpy as np

from .core import Callback, CallbackList


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

    def on_episode_end(self):
        logs = self.callbacks[0].on_episode_end()
        self.history.append(logs)
        return logs

    def on_loop_finish(self, is_aborted):
        self._store()
        return self.callbacks[0].on_loop_finish(is_aborted)

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

    def __init__(self, save_path=None):
        self._reset()

    def on_step_taken(self, transition):
        self.steps += 1
        self.rewards.append(transition.reward)

    def on_episode_end(self):
        logs = {}
        logs["# steps"] = self.steps
        logs["return"] = np.sum(self.rewards)
        logs["max reward"] = np.max(self.rewards)
        logs["min reward"] = np.min(self.rewards)

        self._reset()
        return logs

    def on_loop_finish(self, is_aborted):
        self._reset()

    def _reset(self):
        self.steps = 0
        self.rewards = []
