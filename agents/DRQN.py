from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import random

from humblerl.environments import Transition


class EpisodicMemory(object):

    def __init__(self, buffer_size=10000, trace_length=8, seed=None):
        """Initialize episodic memory.

        Args:
            buffer_size (int): How many episodes to store.
            trace_length (int): How long traces to return while sampling.
            seed (int): Set seed of random module. If None, it'll be seeded from
        current time or from an operating system specific randomness source.
        """
        # Seed random generator.
        # "None or no argument seeds from current time or from an operating
        #  system specific randomness source if available" ~ random doc
        random.seed(seed)

        self._buffer_size = buffer_size
        self._trace_length = trace_length

        # It's before first episode.
        self._episode_idx = 0
        self._episode_buffer = [[], ]

    @staticmethod
    def transition_fabric(state, action, reward, next_state, is_terminal):
        return Transition(state, action, reward, next_state, is_terminal)

    @staticmethod
    def transition_to_list(transition):
        return [transition.state, transition.action,
                transition.reward, transition.next_state]

    def store(self, transition):
        """Store new transition.

        It appends transition to currently started episode.

        Args:
            transition (environments.Transition): Transition packed in namedtuple:
        state, action, reward, next_state, is_terminal.
        """

        transition_list = self.transition_to_list(transition)
        # Check if episode already has been started.
        if len(self._episode_buffer[self._episode_idx]) != 0:
            # Trim current state.
            transition_list = transition_list[1:]

        self._episode_buffer[self._episode_idx] += transition_list

        # After end of episode, prepare for new one.
        if transition.is_terminal:
            self._episode_idx = (self._episode_idx + 1) % self._buffer_size

            assert self._episode_idx <= len(self._episode_buffer), \
                "Illegal episode buffer index!"

            if self._episode_idx == len(self._episode_buffer):
                # Append empty list if there isn't enough items.
                self._episode_buffer.append([])
            else:
                # Clean existing buffer item.
                self._episode_buffer[self._episode_idx] = []

    def sample(self, batch_size):
        """Sample uniformly a batch of episodes traces.

        Args:
            batch_size (int): Size of a batch.

        Returns:
            np.array: Shape (sequence, batch, Transition as list) or
        None if no finished episode.
        """

        # No episode has been fully completed yet.
        if len(self._episode_buffer) <= 1:
            return None

        # Sample batch of traces.
        batch = []
        for _ in range(batch_size):
            batch.append([])

            # Choose episode to sample from.
            episode_idx = random.randrange(len(self._episode_buffer))
            if episode_idx == self._episode_idx:
                episode_idx = (episode_idx + 1) % len(self._episode_buffer)

            # Check if we have enough transition to sample a full trace.
            if len(self._episode_buffer[episode_idx]) // 3 < self._trace_length:
                raise ValueError(
                    "Trace length is too big! Episode {} has only {} transitions."
                    .format(episode_idx, len(self._episode_buffer[episode_idx]) // 3))

            # Each trace consist of 3 items, so episode buffer length divided
            # by 3 will give traces number.
            trace_idx = random.randrange(
                (len(self._episode_buffer[episode_idx]) // 3)
                - self._trace_length + 1
            )

            # Create and stack Transitions.
            for i in range(self._trace_length):
                trace_start = (trace_idx + i) * 3
                # Create list: [state, action, reward, next_state, is_terminal]
                transition_list = \
                    self._episode_buffer[episode_idx][trace_start:trace_start + 4]
                # Check if it is terminal transition
                transition_list.append(
                    trace_idx + i + 1 == len(self._episode_buffer[episode_idx]) // 3)

                # Create Transition and append to batch
                batch[-1].append(self.transition_fabric(*transition_list))

        # Transpose batch to (sequence, batch, Transition) shape.
        sequence = list(map(list, zip(*batch)))

        return np.array(sequence)
