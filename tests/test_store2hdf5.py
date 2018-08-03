import h5py
import numpy as np

from .. import Transition
from ..callbacks import StoreTransitions2Hdf5

HDF5_PATH = "/tmp/test_humblerl_callback.hdf5"


class TestStoreTransitions2Hdf5(object):
    """Test callback on 3D (e.g. images) and continuous states."""

    def test_images_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((8, 8, 3, 2))
        STATE_SPACE[:] = np.array([0, 255])
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreTransitions2Hdf5(ACTION_SPACE, STATE_SPACE, HDF5_PATH,
                                         shuffle=False, min_transitions=MIN_TRANSITIONS,
                                         chunk_size=CHUNK_SIZE, dtype=np.uint8)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                player=idx,
                state=np.random.randint(0, 256, size=(8, 8, 3)),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_player=0,
                next_state=np.random.randint(0, 256, size=(8, 8, 3)),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(transition, None)
        callback.on_loop_finish(False)

        h5py_file = h5py.File(HDF5_PATH, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["ACTION_SPACE"] == ACTION_SPACE)
        assert np.all(h5py_file.attrs["STATE_SPACE"] == STATE_SPACE.shape[:-1])

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)
            assert np.all(h5py_file['next_states'][idx] == transition.next_state)
            assert np.allclose(
                h5py_file['transitions'][idx],
                [transition.player, transition.action, transition.reward, transition.is_terminal]
            )

    def test_continous_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreTransitions2Hdf5(ACTION_SPACE, STATE_SPACE, HDF5_PATH,
                                         shuffle=False, min_transitions=MIN_TRANSITIONS,
                                         chunk_size=CHUNK_SIZE, dtype=np.float)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                player=idx,
                state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_player=0,
                next_state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(transition, None)
        callback.on_loop_finish(False)

        h5py_file = h5py.File(HDF5_PATH, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["ACTION_SPACE"] == ACTION_SPACE)
        assert np.all(h5py_file.attrs["STATE_SPACE"] == STATE_SPACE.shape[:-1])

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)
            assert np.all(h5py_file['next_states'][idx] == transition.next_state)
            assert np.allclose(
                h5py_file['transitions'][idx],
                [transition.player, transition.action, transition.reward, transition.is_terminal]
            )

    def test_shuffle_chunks(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        MIN_TRANSITIONS = 48
        CHUNK_SIZE = 48
        N_TRANSITIONS = 48

        callback = StoreTransitions2Hdf5(ACTION_SPACE, STATE_SPACE, HDF5_PATH,
                                         shuffle=True, min_transitions=MIN_TRANSITIONS,
                                         chunk_size=CHUNK_SIZE, dtype=np.float)

        states = []
        next_states = []
        transitions = []
        for idx in range(N_TRANSITIONS):
            states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            next_states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            transitions.append((idx, np.random.choice(ACTION_SPACE), np.random.normal(0, 1), 0))

            callback.on_step_taken(Transition(
                player=transitions[-1][0],
                state=states[-1],
                action=transitions[-1][1],
                reward=transitions[-1][2],
                next_player=0,
                next_state=next_states[-1],
                is_terminal=transitions[-1][3]
            ), None)

        in_order = True
        h5py_file = h5py.File(HDF5_PATH, "r")
        for idx in range(N_TRANSITIONS):
            state = h5py_file['states'][idx]
            next_state = h5py_file['next_states'][idx]
            transition = h5py_file['transitions'][idx]

            idx_target = states.index(state.tolist())
            if idx != idx_target:
                in_order = False

            assert np.all(h5py_file['states'][idx] == states[idx_target])
            assert np.all(h5py_file['next_states'][idx] == next_states[idx_target])
            assert np.allclose(h5py_file['transitions'][idx], transitions[idx_target])

        assert not in_order, "Data isn't shuffled!"
