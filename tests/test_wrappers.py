import numpy as np

from humblerl import create_gym
from humblerl.environments import Continuous


class TestGymEnvironment(object):
    """Test wrapper on OpenAI Gym toolkit environments."""

    def test_cartpole_env(self):
        """Tests continuous state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("CartPole-v0")

        assert np.allclose(env.state_space, np.array(
            [[-4.8,  4.8], [-3.40282347e+38,  3.40282347e+38],
             [-0.419,  0.419], [-3.40282347e+38,  3.40282347e+38]]), atol=1e-3)
        assert np.all(env.valid_actions == np.array([0, 1]))
        assert env.action_space.num == 2

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)

    def test_frozenlake_env(self):
        """Tests discrete state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("FrozenLake-v0")

        assert np.all(env.state_space == 16)
        assert np.all(env.valid_actions == np.array([0, 1, 2, 3]))
        assert env.action_space.num == 4

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)

    def test_pacman_env(self):
        """Tests box state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("MsPacman-v0")

        assert np.all(env.state_space.shape == (210, 160, 3, 2))
        assert np.all(env.valid_actions == np.array(range(9)))
        assert env.action_space.num == 9

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)

    def test_sokoban_env(self):
        """Tests box state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("Sokoban-v0")

        assert np.all(env.state_space.shape == (160, 160, 3, 2))
        assert np.all(env.valid_actions == np.array(range(8)))
        assert env.action_space.num == 8

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)

    def test_random_maze_env(self):
        """Test random maze environment parameters."""

        env = create_gym("MazeEnv-v0")

        assert np.all(env.state_space.shape == (21, 21, 2))
        assert np.all(env.valid_actions == np.array(range(4)))
        assert env.action_space.num == 4

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)

    def test_mountain_car_continuous_env(self):
        """Tests box state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("MountainCarContinuous-v0")

        assert isinstance(env.action_space, Continuous)
        assert np.all(env.state_space.shape == (2, 2))
        assert np.all(env.valid_actions == np.array(range(1)))
        assert env.action_space.num == 1

        state = env.reset()
        assert np.all(env.current_state == state)

        for action in [[env.action_space.low], [0], [env.action_space.high],
                       env.sample_action()]:
            state, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)
