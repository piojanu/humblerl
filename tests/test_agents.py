import pytest

from humblerl import Vision, ChainVision


class TestVision(object):
    """Test vision preprocessing object."""

    STATE = 7
    REWARD = 666.

    @pytest.fixture
    def double_vision(self):
        return Vision(lambda s: 2 * s,
                      lambda r: 2 * r)

    @pytest.fixture
    def triple_vision(self):
        return Vision(lambda s: 3 * s,
                      lambda r: 3 * r)

    def test_vision(self, double_vision):
        state, reward = double_vision(self.STATE, self.REWARD)

        assert state == 2 * self.STATE
        assert reward == 2 * self.REWARD

    def test_chain_vision(self, double_vision, triple_vision):
        sextuple_vision = ChainVision(double_vision, triple_vision)
        state, reward = sextuple_vision(self.STATE, self.REWARD)

        assert state == 6 * self.STATE
        assert reward == 6 * self.REWARD
