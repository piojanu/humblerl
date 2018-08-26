import csv
import h5py
import numpy as np
import os
import random

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
        logs = self.unwrapped.on_episode_end()
        self.history.append(logs)
        return logs

    def on_loop_finish(self, is_aborted):
        self._store()
        return self.unwrapped.on_loop_finish(is_aborted)

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

    def on_loop_start(self):
        self._reset()

    def on_step_taken(self, transition, info):
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

    def _reset(self):
        self.steps = 0
        self.rewards = []


class StoreTransitions2Hdf5(Callback):
    """Save transitions to HDF5 file."""

    def __init__(self, action_space, state_shape, out_path,
                 shuffle=True, min_transitions=10000, chunk_size=128, dtype=np.uint8):
        """Save transitions to HDF5 file.

        Args:
            action_space (np.ndarray): Action space of Environment.
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
        transition_columns = ["player", "action", "reward", "is_terminal"]

        # Make sure that path to out file exists
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        # Create output hdf5 file and fill metadata
        self.out_file = h5py.File(out_path, "w")
        self.out_file.attrs["N_TRANSITIONS"] = 0
        self.out_file.attrs["N_GAMES"] = 0
        self.out_file.attrs["CHUNK_SIZE"] = chunk_size
        self.out_file.attrs["ACTION_SPACE"] = action_space
        self.out_file.attrs["STATE_SHAPE"] = state_shape

        # Create datasets for states, next_states and transitions
        # NOTE: We save states as np.uint8 dtype because those are RGB pixel values.
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape))
        self.out_next_states = self.out_file.create_dataset(
            name="next_states", dtype=dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape))
        self.out_transitions = self.out_file.create_dataset(
            name="transitions", dtype="f", chunks=(chunk_size, len(transition_columns)),
            shape=(self.dataset_size, len(transition_columns)), maxshape=(None, len(transition_columns)))

        self.transition_count = 0
        self.game_count = 0

        self.states = []
        self.next_states = []
        self.transitions = []

    def on_step_taken(self, transition, info):
        self.states.append(transition.state)
        self.next_states.append(transition.next_state)
        self.transitions.append(
            [transition.player, transition.action, transition.reward, transition.is_terminal])

        if transition.is_terminal:
            self.game_count += 1

        self.transition_count += 1
        if self.transition_count % self.min_transitions == 0:
            self._save_chunk()

    def on_loop_finish(self, is_aborted):
        if len(self.states) > 0:
            self._save_chunk()

        # Close file
        self.out_file.close()

    def _save_chunk(self):
        """Save `states`, `next_states` and `transitions` to HDF5 file. Clear the buffers.
        Update transition and games count in HDF5 file."""

        # Resize datasets if needed
        if self.transition_count > self.dataset_size:
            self.out_states.resize(self.transition_count, axis=0)
            self.out_next_states.resize(self.transition_count, axis=0)
            self.out_transitions.resize(self.transition_count, axis=0)
            self.dataset_size = self.transition_count

        n_transitions = len(self.states)
        start = self.transition_count - n_transitions

        assert n_transitions > 0, "Nothing to save!"

        if self.shuffle_chunk:
            states = []
            next_states = []
            transitions = []

            for idx in random.sample(range(n_transitions), n_transitions):
                states.append(self.states[idx])
                next_states.append(self.next_states[idx])
                transitions.append(self.transitions[idx])
        else:
            states = self.states
            next_states = self.next_states
            transitions = self.transitions

        self.out_states[start:self.transition_count] = \
            np.array(states, dtype=self.state_dtype)
        self.out_next_states[start:self.transition_count] = \
            np.array(next_states, dtype=self.state_dtype)
        self.out_transitions[start:self.transition_count] = transitions

        self.out_file.attrs["N_TRANSITIONS"] = self.transition_count
        self.out_file.attrs["N_GAMES"] = self.game_count

        self.states = []
        self.next_states = []
        self.transitions = []
