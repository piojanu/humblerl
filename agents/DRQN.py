import random


class EpisodicMemory(object):
    def __init__(self, buffer_size=1000, trace_length=8, seed=None):
        """Initialize episodic memory.

        Args:
            buffer_size (int): How many episodes to store.
            trace_length (int): How long traces to return while sampling.
            seed (int): Set seed of random module. If None, it'll be set to 
        """
        # Seed random generator.
        # "None or no argument seeds from current time or from an operating
        #  system specific randomness source if available" ~ random doc
        random.seed(seed)

        self._buffer_size = buffer_size
        self._trace_length = trace_length

        # It's before first episode.
        self._buffer_idx = -1
        self._episode_buffer = []

    def start(self, state):
        """Starts new episode.

        Args:
            state (np.array): Initial state.
        """

        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size

        # Append empty list if there isn't enough items.
        if self._buffer_idx == len(self._episode_buffer):
            self._episode_buffer.append([])

        assert self._buffer_idx <= len(self._episode_buffer), \
            "Illegal episode buffer index!"

        self._episode_buffer[self._buffer_idx] = [state, ]

    def store(self, action, reward, state):
        """Store new transition.

        It appends transition to currently started episode.

        Args:
            action (list of floats): Action performed in this transition.
            reward (float): Gathered reward in this transition.
            state (np.array): Observation/state at the end of transition.
        """

        # Check if any episode has been started.
        if len(self._episode_buffer) == 0:
            raise ValueError("No episode started!")

        self._episode_buffer[self._buffer_idx] += [action, reward, state]

    def sample(self, batch_size):
        """Sample uniformly a batch of episode traces.

        Args:
            batch_size (int): Size of a batch.

        Returns:
            list: Batch of transitions, shape: [batch_size, trace_length, 4] or
        None if no finished episode.
        """

        if len(self._episode_buffer) <= 1:
            return None

        batch = []
        for _ in range(batch_size):
            batch.append([])

            episode_idx = random.randrange(len(self._episode_buffer))
            if episode_idx == self._buffer_idx:
                episode_idx = (episode_idx + 1) % self._buffer_size

            if len(self._episode_buffer[episode_idx]) // 3 > self._trace_length:
                raise ValueError("Trace length is too big! Episode {} \
                                 has only {} traces.".format(
                    episode_idx, len(self._episode_buffer[episode_idx])))

            # Each trace consist of 3 items, so episode buffer length divided
            # by 3 will give traces number.
            trace_idx = random.randrange(
                (len(self._episode_buffer[episode_idx]) // 3)
                - self._trace_length + 1
            )

            # Build and stack transitions
            for i in range(self._trace_length):
                trace_start = (trace_idx + i) * 3
                # Append transition: (state before, action, reward, state after)
                batch[-1].append(self._episode_buffer[episode_idx]
                                 [trace_start:trace_start + 4])

        return batch
