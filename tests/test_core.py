import numpy as np
import pytest

from .. import ply, loop, Callback, Environment, Mind, Transition
from unittest.mock import MagicMock, PropertyMock


class TestBasicCore(object):
    """Test loop and ply functions."""

    INIT_STATE = 'init_state'
    NEXT_STATE = 'next_state'
    REWARD = 0
    CURR_PLAYER = 0
    PLAYER_NUMBER = 1
    VALID_ACTIONS = np.array([0, 1])
    LOGITS = np.array([0.5, 1.6])

    @pytest.fixture
    def env(self):
        mock = MagicMock(spec=Environment)
        mock.reset.return_value = self.INIT_STATE, self.CURR_PLAYER
        mock.step.return_value = self.NEXT_STATE, self.CURR_PLAYER, self.REWARD, True, None
        type(mock).current_player = PropertyMock(return_value=self.CURR_PLAYER)
        type(mock).current_state = PropertyMock(return_value=self.INIT_STATE)
        type(mock).players_number = PropertyMock(return_value=self.PLAYER_NUMBER)
        type(mock).valid_actions = PropertyMock(return_value=self.VALID_ACTIONS)

        return mock

    @pytest.fixture
    def mind(self):
        mock = MagicMock(spec=Mind)
        mock.plan.return_value = self.LOGITS

        return mock

    @pytest.fixture
    def callback(self):
        mock = MagicMock(spec=Callback)
        type(mock).metrics = PropertyMock(return_value={})

        return mock

    def test_ply(self, env, mind, callback):
        train_mode = True
        debug_mode = False
        transition = Transition(self.CURR_PLAYER, self.INIT_STATE, np.argmax(self.VALID_ACTIONS),
                                self.REWARD, self.CURR_PLAYER, self.NEXT_STATE, True)

        ply(env, mind, policy='deterministic',
            train_mode=train_mode, debug_mode=debug_mode, callbacks=[callback])

        env.step.assert_called_once_with(np.argmax(self.VALID_ACTIONS))
        mind.plan.assert_called_once_with(self.INIT_STATE, self.CURR_PLAYER, train_mode, debug_mode)
        callback.on_action_planned.assert_called_once_with(0, self.LOGITS, None)
        callback.on_step_taken.assert_called_once_with(0, transition, None)

    def test_loop(self, env, mind, callback):
        train_mode = False

        loop(env, mind, train_mode=train_mode, verbose=0, callbacks=[callback])

        env.reset.assert_called_once_with(train_mode, first_player=self.CURR_PLAYER)
        callback.on_loop_start.assert_called_once_with()
        callback.on_episode_start.assert_called_once_with(0, train_mode)
        callback.on_episode_end.assert_called_once_with(0, train_mode)
        callback.on_loop_end.assert_called_once_with(False)
