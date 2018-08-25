import numpy as np

from .. import create_gym


class TestGymEnvironment(object):
    """Test wrapper on OpenAI Gym toolkit environments."""

    def test_cartpole_env(self):
        """Tests continuous state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("CartPole-v0")

        assert env.players_number == 1
        assert np.allclose(env.state_space, np.array(
            [[-4.8,  4.8], [-3.40282347e+38,  3.40282347e+38],
             [-0.419,  0.419], [-3.40282347e+38,  3.40282347e+38]]), atol=1e-3)
        assert np.all(env.valid_actions == np.array([0, 1]))

        state, _ = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, next_player, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)
            assert next_player == 0

    def test_frozenlake_env(self):
        """Tests discrete state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("FrozenLake-v0")

        assert env.players_number == 1
        assert np.all(env.state_space == np.array([16]))
        assert np.all(env.valid_actions == np.array([0, 1, 2, 3]))

        state, _ = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, next_player, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)
            assert next_player == 0

    def test_pacman_env(self):
        """Tests box state space and discrete action space handling,
        all properties, reset, step and create_gym methods."""

        env = create_gym("MsPacman-v0")

        assert env.players_number == 1
        assert np.all(env.state_space.shape == (210, 160, 3, 2))
        assert np.all(env.valid_actions == np.array(range(9)))

        state, _ = env.reset()
        assert np.all(env.current_state == state)

        for action in env.valid_actions:
            state, next_player, reward, done, info = env.step(action)
            assert np.all(env.current_state == state)
            assert next_player == 0
