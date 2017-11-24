from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from mock import MagicMock
import pytest

from humblerl.agents import Agent, Vision
from humblerl.environments import Environment


class TestVision(object):
    MOCK_STATE = [1, 2, 3]
    MOCK_REWARD = 7.

    def test_default_vision_system(self):
        vision_system = Vision()
        state, reward = vision_system(self.MOCK_STATE, self.MOCK_REWARD)

        assert state == self.MOCK_STATE
        assert reward == self.MOCK_REWARD

    def test_custom_vision_system(self):
        def state_processor(x):
            return x + ["dupa"]

        def reward_processor(x):
            return x * 2

        vision_system = Vision(state_processor, reward_processor)
        state, reward = vision_system(self.MOCK_STATE, self.MOCK_REWARD)

        assert state == state_processor(self.MOCK_STATE)
        assert reward == reward_processor(self.MOCK_REWARD)


class TestAgent(object):
    _MOCK_INIT_STATE = [1, 2, 3]
    _MOCK_ACTION = [7., ]
    _MOCK_INFO = "Dupa22"
    _MOCK_NEXT_STATE = [4, 5, 6]
    _MOCK_REWARD = 7.
    _MOCK_DONE = False

    _MOCK_VISION_STATE = [7, 8, 9]
    _MOCK_VISION_REWARD = 9.

    @staticmethod
    def mock_policy(x):
        return (TestAgent._MOCK_ACTION, TestAgent._MOCK_INFO)

    @pytest.fixture
    def agent_mock_env(self):
        env = MagicMock(spec=Environment)
        env.reset.return_value = self._MOCK_INIT_STATE
        env.step.return_value = (self._MOCK_NEXT_STATE,
                                 self._MOCK_REWARD,
                                 self._MOCK_DONE)

        return (Agent(env=env), env)

    @pytest.fixture
    def agent_mock_env_and_vision(self):
        env = MagicMock(spec=Environment)
        env.reset.return_value = self._MOCK_INIT_STATE
        env.step.return_value = (self._MOCK_NEXT_STATE,
                                 self._MOCK_REWARD,
                                 self._MOCK_DONE)

        vision = MagicMock(spec=Vision)
        vision.return_value = (
            self._MOCK_VISION_STATE, self._MOCK_VISION_REWARD)

        return (Agent(env=env, vision=vision), env, vision)

    def test_reset(self, agent_mock_env):
        agent, env = agent_mock_env

        state = agent.reset(train_mode=False)

        assert state == self._MOCK_INIT_STATE
        assert agent._cur_state == self._MOCK_INIT_STATE
        env.reset.assert_called_with(train_mode=False)

    def test_step(self, agent_mock_env):
        agent, env = agent_mock_env

        agent.reset()
        transition, info = agent.step(
            policy=self.mock_policy
        )

        assert info == self._MOCK_INFO
        assert transition.state == self._MOCK_INIT_STATE
        assert transition.action == self._MOCK_ACTION
        assert transition.reward == self._MOCK_REWARD
        assert transition.next_state == self._MOCK_NEXT_STATE
        assert transition.is_terminal == self._MOCK_DONE
        assert agent._cur_policy == self.mock_policy
        env.step.assert_called_with(action=self._MOCK_ACTION)

    def test_run(self, agent_mock_env):
        agent, env = agent_mock_env

        agent.reset()
        agent.policy = self.mock_policy

        stop = 3
        step = 0

        for transition, info in agent.run(stop):
            assert info == self._MOCK_INFO
            assert (transition.state == self._MOCK_INIT_STATE or
                    transition.state == self._MOCK_NEXT_STATE)
            assert transition.action == self._MOCK_ACTION
            assert transition.reward == self._MOCK_REWARD
            assert transition.next_state == self._MOCK_NEXT_STATE
            assert transition.is_terminal == self._MOCK_DONE
            assert agent._cur_policy == self.mock_policy
            env.step.assert_called_with(action=self._MOCK_ACTION)

            step += 1

        assert step == stop

    def test_run_early_stop(self, mocker, agent_mock_env):
        agent, env = agent_mock_env
        env.step.return_value = (self._MOCK_NEXT_STATE,
                                 self._MOCK_REWARD,
                                 True)

        agent.reset()
        agent.policy = self.mock_policy

        stop = 3
        step = 0

        for transition, info in agent.run(stop):
            assert info == self._MOCK_INFO
            assert transition.state == self._MOCK_INIT_STATE
            assert transition.action == self._MOCK_ACTION
            assert transition.reward == self._MOCK_REWARD
            assert transition.next_state == self._MOCK_NEXT_STATE
            assert transition.is_terminal == True
            assert agent._cur_policy == self.mock_policy
            env.step.assert_called_with(action=self._MOCK_ACTION)

            step += 1

        assert step == 1

    def test_step_without_policy(self, agent_mock_env):
        agent, env = agent_mock_env

        with pytest.raises(ValueError):
            agent.reset()
            agent.step()

    def test_step_without_reset(self, agent_mock_env):
        def policy(x): return self._MOCK_ACTION
        agent, env = agent_mock_env

        with pytest.raises(ValueError):
            agent.step(policy=policy)

    def test_vision_reset(self, agent_mock_env_and_vision):
        agent, env, vision = agent_mock_env_and_vision

        state = agent.reset(train_mode=False)

        assert state == self._MOCK_VISION_STATE
        assert agent._cur_state == self._MOCK_VISION_STATE
        env.reset.assert_called_with(train_mode=False)
        vision.assert_called_with(self._MOCK_INIT_STATE, 0)

    def test_vision_step(self, agent_mock_env_and_vision):
        agent, env, vision = agent_mock_env_and_vision

        agent.reset()
        transition, info = agent.step(
            policy=self.mock_policy
        )

        assert info == self._MOCK_INFO
        assert transition.state == self._MOCK_VISION_STATE
        assert transition.action == self._MOCK_ACTION
        assert transition.reward == self._MOCK_VISION_REWARD
        assert transition.next_state == self._MOCK_VISION_STATE
        assert transition.is_terminal == self._MOCK_DONE
        assert agent._cur_policy == self.mock_policy
        env.step.assert_called_with(action=self._MOCK_ACTION)
        vision.assert_called_with(self._MOCK_NEXT_STATE, self._MOCK_REWARD)
