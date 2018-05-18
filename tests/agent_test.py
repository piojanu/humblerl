from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from mock import MagicMock, PropertyMock
import pytest

from humblerl import Agent, Vision, Policy
from humblerl import Environment


class TestVision(object):
    MOCK_STATE = [1, 2, 3]
    MOCK_REWARD = 7.

    def test_default_vision_system(self):
        vision_system = Vision()
        state, reward = vision_system(self.MOCK_STATE, self.MOCK_REWARD)

        assert state == self.MOCK_STATE
        assert reward == self.MOCK_REWARD

    def test_default_vision_system_only_state(self):
        vision_system = Vision()
        state = vision_system(self.MOCK_STATE)

        assert state == self.MOCK_STATE

    def test_custom_vision_system(self):
        def state_processor(x):
            return x + ["dupa"]

        def reward_processor(x):
            return x * 2

        vision_system = Vision(state_processor, reward_processor)
        state, reward = vision_system(self.MOCK_STATE, self.MOCK_REWARD)

        assert state == state_processor(self.MOCK_STATE)
        assert reward == reward_processor(self.MOCK_REWARD)

    def test_custom_vision_system_only_state(self):
        def state_processor(x):
            return x + ["dupa"]

        vision_system = Vision(state_processor)
        state = vision_system(self.MOCK_STATE)

        assert state == state_processor(self.MOCK_STATE)


class TestAgent(object):
    _MOCK_INIT_STATE = [1, 2, 3]
    _MOCK_ACTION = [7., ]
    _MOCK_NEXT_STATE = [4, 5, 6]
    _MOCK_REWARD = 7.
    _MOCK_DONE = False

    _MOCK_VISION_STATE = [7, 8, 9]
    _MOCK_VISION_REWARD = 9.

    @pytest.fixture
    def agent_mock(self):
        env = MagicMock(spec=Environment)

        def env_reset(train_mode):
            type(env).current_state = PropertyMock(return_value=self._MOCK_INIT_STATE)
            return self._MOCK_INIT_STATE

        def env_step(action):
            type(env).current_state = PropertyMock(return_value=self._MOCK_NEXT_STATE)
            return (self._MOCK_NEXT_STATE, self._MOCK_REWARD, self._MOCK_DONE)

        type(env).current_state = PropertyMock(return_value=None)
        env.reset.side_effect = env_reset
        env.step.side_effect = env_step

        model = MagicMock(spec=Policy)
        model.select_action.return_value = self._MOCK_ACTION

        return (Agent(env=env, model=model), env, model)

    @pytest.fixture
    def agent_mock_with_vision(self):
        env = MagicMock(spec=Environment)

        def env_reset(train_mode):
            type(env).current_state = PropertyMock(return_value=self._MOCK_INIT_STATE)
            return self._MOCK_INIT_STATE

        def env_step(action):
            type(env).current_state = PropertyMock(return_value=self._MOCK_NEXT_STATE)
            return (self._MOCK_NEXT_STATE, self._MOCK_REWARD, self._MOCK_DONE)

        type(env).current_state = PropertyMock(return_value=None)
        env.reset.side_effect = env_reset
        env.step.side_effect = env_step

        vision = Vision(lambda s: self._MOCK_VISION_STATE, lambda r: self._MOCK_VISION_REWARD)

        model = MagicMock(spec=Policy)
        model.select_action.return_value = self._MOCK_ACTION

        return (Agent(env=env, model=model, vision=vision), env, model, vision)

    @staticmethod
    def transition_env_model_check(transition, env, model, step=0, is_model_used=True,
                                   mock_init_state=_MOCK_INIT_STATE,
                                   mock_action=_MOCK_ACTION,
                                   mock_next_state=_MOCK_NEXT_STATE,
                                   mock_reward=_MOCK_REWARD,
                                   mock_done=_MOCK_DONE):

        def check_model(mock_state):
            assert transition.state == mock_state
            if is_model_used:
                model.select_action.assert_called_with(curr_state=mock_state)
            else:
                model.select_action.assert_not_called()

        assert transition.action == mock_action
        assert transition.reward == mock_reward
        assert transition.next_state == mock_next_state
        assert transition.is_terminal == mock_done
        env.step.assert_called_with(action=mock_action)
        if step == 0:
            check_model(mock_init_state)
        else:
            check_model(mock_next_state)
        model.report.assert_called_with(transition=transition)

    def test_reset(self, agent_mock):
        agent, env, model = agent_mock

        state = agent.reset(train_mode=False)

        assert state == self._MOCK_INIT_STATE
        env.reset.assert_called_with(train_mode=False)

    def test_do_action(self, agent_mock):
        agent, env, model = agent_mock

        agent.reset()
        transition = agent.do(
            self._MOCK_ACTION
        )

        self.transition_env_model_check(transition, env, model, is_model_used=False)

    def test_do_model(self, agent_mock):
        agent, env, model = agent_mock

        agent.reset()
        transition = agent.do()

        self.transition_env_model_check(transition, env, model)

    def test_play(self, agent_mock):
        agent, env, model = agent_mock

        agent.reset()

        stop = 3
        step = 0

        for transition in agent.play(stop):
            self.transition_env_model_check(transition, env, model, step=step)

            step += 1

        assert step == stop

    def test_play_early_stop(self, agent_mock):
        agent, env, model = agent_mock

        def env_step(action):
            type(env).current_state = PropertyMock(return_value=self._MOCK_NEXT_STATE)
            return (self._MOCK_NEXT_STATE, self._MOCK_REWARD, True)
        env.step.side_effect = env_step

        agent.reset()

        stop = 3
        step = 0

        for transition in agent.play(stop):
            self.transition_env_model_check(transition, env, model, step=step, mock_done=True)

            step += 1

        assert step == 1

    def test_step_without_reset(self, agent_mock):
        agent, env, model = agent_mock

        with pytest.raises(ValueError) as error:
            agent.do(self._MOCK_ACTION)
        assert "You need to reset agent first!" \
            in str(error.value)

    def test_vision_reset(self, agent_mock_with_vision):
        agent, env, model, vision = agent_mock_with_vision

        state = agent.reset(train_mode=False)

        assert state == self._MOCK_VISION_STATE
        env.reset.assert_called_with(train_mode=False)

    def test_vision_do(self, agent_mock_with_vision):
        agent, env, model, vision = agent_mock_with_vision

        agent.reset()
        transition = agent.do(
            self._MOCK_ACTION
        )

        self.transition_env_model_check(transition, env, model, is_model_used=False,
                                        mock_init_state=self._MOCK_VISION_STATE,
                                        mock_next_state=self._MOCK_VISION_STATE,
                                        mock_reward=self._MOCK_VISION_REWARD)
