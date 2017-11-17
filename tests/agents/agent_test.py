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
    _MOCK_NEXT_STATE = [4, 5, 6]
    _MOCK_REWARD = 7.
    _MOCK_DONE = True

    _MOCK_VISION_STATE = [7, 8, 9]
    _MOCK_VISION_REWARD = 9.

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
        def policy(x): return self._MOCK_ACTION
        agent, env = agent_mock_env

        agent.reset()
        action, state, reward, done = agent.step(
            policy=policy
        )

        assert action == self._MOCK_ACTION
        assert state == self._MOCK_NEXT_STATE
        assert reward == self._MOCK_REWARD
        assert done == self._MOCK_DONE
        assert agent._cur_policy == policy
        env.step.assert_called_with(action=self._MOCK_ACTION)

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
        def policy(x): return self._MOCK_ACTION
        agent, env, vision = agent_mock_env_and_vision

        agent.reset()
        action, state, reward, done = agent.step(
            policy=policy
        )

        assert action == self._MOCK_ACTION
        assert state == self._MOCK_VISION_STATE
        assert reward == self._MOCK_VISION_REWARD
        assert done == self._MOCK_DONE
        assert agent._cur_policy == policy
        env.step.assert_called_with(action=self._MOCK_ACTION)
        vision.assert_called_with(self._MOCK_NEXT_STATE, self._MOCK_REWARD)
