import csv
import h5py
import numpy as np
import os
import random

from abc import ABCMeta, abstractmethod


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


class StoreStates2Hdf5(Callback):
    """Save transitions to HDF5 file in three datasets:
        * 'states': Keeps transition's state (e.g. image).
        Datasets are organized in such a way, that you can access transition 'I' by accessing
        'I'-th position in all three datasets.

        HDF5 file also keeps meta-informations (attributes) as such:
        * 'N_TRANSITIONS': Datasets size (number of transitions).
        * 'N_GAMES': From how many games those transitions come from.
        * 'CHUNK_SIZE': HDF5 file chunk size (batch size should be multiple of it).
        * 'STATE_SHAPE': Shape of environment's state ('(next_)states' dataset element shape).
    """

    def __init__(self, state_shape, out_path, shuffle=True, min_transitions=10000, chunk_size=128, dtype=np.uint8):
        """Save transitions to HDF5 file.

        Args:
            state_shape (tuple): Shape of environment's state.
            out_path (str): Path to HDF5 file where transition will be stored.
            shuffle (bool): If data should be shuffled (in subsets of `min_transitions` number of
                transitions). (Default: True)
            min_transitions (int): Minimum size of dataset in transitions number. Also, whenever
                this amount of transitions is gathered, data is shuffled (if requested) and stored
                on disk. (Default: 10000)
            chunk_size (int): Chunk size in transitions. For efficiency reasons, data is saved
                to file in chunks to limit the disk usage (chunk is smallest unit that get fetched
                from disk). For best performance set it to training batch size and in e.g. Keras
                use shuffle='batch'/False. Never use shuffle=True, as random access to hdf5 is
                slow. (Default: 128)
            dtype (np.dtype): Data type of states. (Default: np.uint8)
        """

        self.out_path = out_path
        self.dataset_size = min_transitions
        self.shuffle_chunk = shuffle
        self.min_transitions = min_transitions
        self.state_dtype = dtype

        # Make sure that path to out file exists
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        # Create output hdf5 file and fill metadata
        self.out_file = h5py.File(out_path, "w")
        self.out_file.attrs["N_TRANSITIONS"] = 0
        self.out_file.attrs["N_GAMES"] = 0
        self.out_file.attrs["CHUNK_SIZE"] = chunk_size
        self.out_file.attrs["STATE_SHAPE"] = state_shape

        # Create dataset for states
        # NOTE: We save states as np.uint8 dtype because those are RGB pixel values.
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape),
            compression="lzf")

        self.transition_count = 0
        self.game_count = 0

        self.states = []

    def on_step_taken(self, step, transition, info):
        self.states.append(transition.state)

        if transition.is_terminal:
            self.game_count += 1

        self.transition_count += 1
        if self.transition_count % self.min_transitions == 0:
            self._save_chunk()

    def on_loop_end(self, is_aborted):
        if len(self.states) > 0:
            self._save_chunk()

        # Close file
        self.out_file.close()

    def _save_chunk(self):
        """Save `states`  to HDF5 file. Clear the buffers.
        Update transition and games count in HDF5 file."""

        # Resize datasets if needed
        if self.transition_count > self.dataset_size:
            self.out_states.resize(self.transition_count, axis=0)
            self.dataset_size = self.transition_count

        n_transitions = len(self.states)
        start = self.transition_count - n_transitions

        assert n_transitions > 0, "Nothing to save!"

        if self.shuffle_chunk:
            states = []

            for idx in random.sample(range(n_transitions), n_transitions):
                states.append(self.states[idx])
        else:
            states = self.states

        self.out_states[start:self.transition_count] = \
            np.array(states, dtype=self.state_dtype)

        self.out_file.attrs["N_TRANSITIONS"] = self.transition_count
        self.out_file.attrs["N_GAMES"] = self.game_count

        self.states = []
