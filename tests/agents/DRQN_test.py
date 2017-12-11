from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pytest
import random

from humblerl.agents import EpisodicMemory
from humblerl.environments import Transition


class TestEpisodicMemory(object):

    @staticmethod
    def transition_fabric(
            state=[1, 2, 3],
            action=[7., ],
            reward=7.,
            next_state=[4, 5, 6],
            is_terminal=False):
        return Transition(state, action, reward, next_state, is_terminal)

    @staticmethod
    def transition_to_list(transition):
        return [transition.state, transition.action,
                transition.reward, transition.next_state]

    @pytest.fixture
    def transition(self):
        return self.transition_fabric()

    @pytest.fixture
    def episodicmemory(self):
        episodicmemory = EpisodicMemory(
            buffer_size=10,
            trace_length=8,
            seed=123)

        return episodicmemory

    def test_store(self, episodicmemory, transition):
        episodicmemory.store(transition)

        assert episodicmemory._episode_buffer[0] == \
            self.transition_to_list(transition)

    def test_stress_store(self, episodicmemory):
        transition = self.transition_fabric(is_terminal=True)

        # Make up buffer to the full
        for i in range(episodicmemory._buffer_size):
            episodicmemory.store(transition)
            assert episodicmemory._episode_buffer[i] == \
                self.transition_to_list(transition)

        # Add one more transition and check if it's at the beginning
        transition = self.transition_fabric(state=["dupa", ], is_terminal=True)
        episodicmemory.store(transition)

        assert episodicmemory._episode_buffer[0] == \
            self.transition_to_list(transition)

    @pytest.fixture
    def filled_episodicmemory(self):
        random.seed(123)
        TRACE_LENGTH = 7

        episodicmemory = EpisodicMemory(
            buffer_size=100,
            trace_length=TRACE_LENGTH,
            seed=123)

        targets = []
        current_state = [random.randrange(100) for _ in range(5)]
        for i in range(TRACE_LENGTH):
            transition = \
                self.transition_fabric(
                    state=current_state,
                    action=[random.random(), ],
                    reward=float(random.randrange(-10, 10)),
                    next_state=[random.randrange(100) for _ in range(5)],
                    is_terminal=(i == TRACE_LENGTH - 1))

            current_state = transition.next_state

            episodicmemory.store(transition)
            targets.append(transition)

        return (episodicmemory, np.array(targets))

    def test_sample_before_store(self, episodicmemory):
        assert episodicmemory.sample(1) == None

    def test_sample_before_episode_end(self, episodicmemory):
        # Store one non terminal transition.
        episodicmemory.store(self.transition_fabric())

        assert episodicmemory.sample(1) == None

    def test_sample(self, filled_episodicmemory):
        BATCH_SIZE = 3

        episodicmemory, targets = filled_episodicmemory
        sequence = episodicmemory.sample(BATCH_SIZE)

        assert len(sequence) == len(targets)
        for batch, target in zip(sequence, targets):
            assert len(batch) == BATCH_SIZE
            for transition in batch:
                assert np.array_equal(transition, target)

    def test_too_few_transitions(self, episodicmemory):
        # Store not enough transitions for one trace.
        episodicmemory.store(self.transition_fabric(is_terminal=True))

        with pytest.raises(ValueError) as error:
            episodicmemory.sample(1)
        assert "Trace length is too big! Episode 0 has only 1 transitions." \
            in str(error.value)