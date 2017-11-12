import pytest

from humblerl.agents import EpisodicMemory


class TestEpisodicMemoryStart(object):

    @pytest.fixture
    def episodicmemory(self):
        BUFFER_SIZE = 100
        TRACE_LENGTH = 8
        RANDOM_SEED = 123
        return EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=TRACE_LENGTH,
            seed=RANDOM_SEED)

    def test_valid_start(self, episodicmemory):
        INIT_STATE = [1, 2, 3]

        episodicmemory.start(INIT_STATE)

        assert episodicmemory._buffer_idx == 0
        assert len(episodicmemory._episode_buffer) == 1
        assert episodicmemory._episode_buffer[0][0] == INIT_STATE

    def test_start_after_start(self, episodicmemory):
        INIT_STATE = [1, 2, 3]
        INIT_STATE_TWO = [7, 8, 9]

        episodicmemory.start(INIT_STATE)
        episodicmemory.start(INIT_STATE_TWO)

        assert episodicmemory._buffer_idx == 1
        assert len(episodicmemory._episode_buffer) == 2
        assert episodicmemory._episode_buffer[0][0] == INIT_STATE
        assert episodicmemory._episode_buffer[1][0] == INIT_STATE_TWO

    def test_corrupt_start(self, episodicmemory):
        episodicmemory._buffer_idx = 0

        with pytest.raises(AssertionError):
            episodicmemory.start([])


class TestEpisodicMemoryStore(object):
    INIT_STATE = [1, 2, 3]

    @pytest.fixture
    def episodicmemory(self):
        BUFFER_SIZE = 100
        TRACE_LENGTH = 8
        RANDOM_SEED = 123
        return EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=TRACE_LENGTH,
            seed=RANDOM_SEED)

    @pytest.fixture
    def started_episodicmemory(self):
        BUFFER_SIZE = 100
        TRACE_LENGTH = 8
        RANDOM_SEED = 123

        episodicmemory = EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=TRACE_LENGTH,
            seed=RANDOM_SEED)
        episodicmemory.start(self.INIT_STATE)

        return episodicmemory

    def test_store_before_start(self, episodicmemory):
        with pytest.raises(ValueError):
            episodicmemory.store([], 0., [])

    def test_store_after_start(self, started_episodicmemory):
        NEXT_TRACE = [[0, 1], 7., [4, 5, 6]]
        TARGET_TRANSITION = [self.INIT_STATE, ] + NEXT_TRACE

        started_episodicmemory.store(*NEXT_TRACE)

        assert started_episodicmemory._episode_buffer[0] == TARGET_TRANSITION


class TestEpisodicMemorySample(object):
    INIT_STATE = [1, 2, 3]
    NEXT_TRACE = [[0, 1], 7., [4, 5, 6]]
    TRACE_LENGTH = 2

    @pytest.fixture
    def episodicmemory(self):
        BUFFER_SIZE = 100
        RANDOM_SEED = 123
        return EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=self.TRACE_LENGTH,
            seed=RANDOM_SEED)

    @pytest.fixture
    def started_episodicmemory(self):
        BUFFER_SIZE = 100
        RANDOM_SEED = 123

        episodicmemory = EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=self.TRACE_LENGTH,
            seed=RANDOM_SEED)
        episodicmemory.start(self.INIT_STATE)

        return episodicmemory

    @pytest.fixture
    def filled_episodicmemory(self):
        BUFFER_SIZE = 100
        RANDOM_SEED = 123

        episodicmemory = EpisodicMemory(
            buffer_size=BUFFER_SIZE,
            trace_length=self.TRACE_LENGTH,
            seed=RANDOM_SEED)
        episodicmemory.start(self.INIT_STATE)

        for _ in range(self.TRACE_LENGTH):
            episodicmemory.store(*self.NEXT_TRACE)

        episodicmemory.start([])

        return episodicmemory

    def test_sample_before_start(self, episodicmemory):
        assert episodicmemory.sample(1) == None

    def test_sample_before_store(self, started_episodicmemory):
        assert started_episodicmemory.sample(1) == None

    def test_valid_sample(self, filled_episodicmemory):
        FIRST_TRANSITION = [self.INIT_STATE, ] + self.NEXT_TRACE
        SECOND_TRANSITION = [self.NEXT_TRACE[2], ] + self.NEXT_TRACE

        batch = filled_episodicmemory.sample(2)

        assert len(batch) == 2
        assert len(batch[0]) == 2
        assert batch[0][0] == FIRST_TRANSITION
        assert batch[0][1] == SECOND_TRANSITION
        assert batch[1][0] == FIRST_TRANSITION
        assert batch[1][1] == SECOND_TRANSITION

    def test_corrupt_sample(self, filled_episodicmemory):
        filled_episodicmemory._trace_length = self.TRACE_LENGTH + 1

        with pytest.raises(ValueError):
            filled_episodicmemory.sample(1)
