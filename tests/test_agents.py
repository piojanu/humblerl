import pytest

from humblerl import Interpreter, ChainInterpreter


class TestInterpreter(object):
    """Test interpreter preprocessing object."""

    STATE = 7
    REWARD = 666.

    @pytest.fixture
    def double_interpreter(self):
        return Interpreter(lambda s: 2 * s,
                      lambda r: 2 * r)

    @pytest.fixture
    def triple_interpreter(self):
        return Interpreter(lambda s: 3 * s,
                      lambda r: 3 * r)

    def test_interpreter(self, double_interpreter):
        state, reward = double_interpreter(self.STATE, self.REWARD)

        assert state == 2 * self.STATE
        assert reward == 2 * self.REWARD

    def test_chain_interpreter(self, double_interpreter, triple_interpreter):
        sextuple_interpreter = ChainInterpreter(double_interpreter, triple_interpreter)
        state, reward = sextuple_interpreter(self.STATE, self.REWARD)

        assert state == 6 * self.STATE
        assert reward == 6 * self.REWARD
