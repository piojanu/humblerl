import h5py
import numpy as np

from humblerl import Transition
from humblerl.callbacks import StoreStates2Hdf5

HDF5_PATH = "/tmp/test_humblerl_callback.hdf5"


class TestStoreTransitions2Hdf5(object):
    """Test callback on 3D (e.g. images) and continuous states."""

    def test_images_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((8, 8, 3, 2))
        STATE_SPACE[:] = np.array([0, 255])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreStates2Hdf5(STATE_SPACE_SHAPE, HDF5_PATH,
                                    shuffle=False, min_transitions=MIN_TRANSITIONS,
                                    chunk_size=CHUNK_SIZE, dtype=np.uint8)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                state=np.random.randint(0, 256, size=(8, 8, 3)),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_state=np.random.randint(0, 256, size=(8, 8, 3)),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(HDF5_PATH, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == STATE_SPACE_SHAPE)

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)

    def test_continous_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreStates2Hdf5(STATE_SPACE_SHAPE, HDF5_PATH,
                                    shuffle=False, min_transitions=MIN_TRANSITIONS,
                                    chunk_size=CHUNK_SIZE, dtype=np.float)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(HDF5_PATH, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == STATE_SPACE_SHAPE)

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)

    def test_shuffle_chunks(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 48
        CHUNK_SIZE = 48
        N_TRANSITIONS = 48

        callback = StoreStates2Hdf5(STATE_SPACE_SHAPE, HDF5_PATH,
                                    shuffle=True, min_transitions=MIN_TRANSITIONS,
                                    chunk_size=CHUNK_SIZE, dtype=np.float)

        states = []
        next_states = []
        transitions = []
        for idx in range(N_TRANSITIONS):
            states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            next_states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            transitions.append((np.random.choice(ACTION_SPACE), np.random.normal(0, 1), 0))

            callback.on_step_taken(idx, Transition(
                state=states[-1],
                action=transitions[-1][0],
                reward=transitions[-1][1],
                next_state=next_states[-1],
                is_terminal=transitions[-1][2]
            ), None)

        in_order = True
        h5py_file = h5py.File(HDF5_PATH, "r")
        for idx in range(N_TRANSITIONS):
            state = h5py_file['states'][idx]

            idx_target = states.index(state.tolist())
            if idx != idx_target:
                in_order = False

            assert np.all(h5py_file['states'][idx] == states[idx_target])

        assert not in_order, "Data isn't shuffled!"
