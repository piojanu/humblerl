from abc import ABCMeta, abstractmethod


class Callback(metaclass=ABCMeta):
    """Abstract class responsible for event handling."""

    @abstractmethod
    def report_step(self, transition):
        """Receives step report from Agent.

        Args:
          transition (Transition): Last transition (state, action, reward, next state, is terminal)
        in RL environment.
        """
        pass
