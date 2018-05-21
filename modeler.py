from abc import ABCMeta, abstractmethod

class Dynamics(metaclass=ABCMeta):
    """Interface to dynamics model."""

    @abstractmethod
    def __call__(self, state, action):
        """Simulate a world transition.

        Args:
          state (object): Current world state to start from.
          action (object): Action to take in current world state.
        None action is not allowed.
        
        Returns:
          (object): Next state after taking the given action in the given state.
        If action is not allowed, returns None.
        """
        pass


class PerfectDynamics(Dynamics, metaclass=ABCMeta):
    """Interface to perfect information dynamics model."""

    @abstractmethod
    def __call__(self, state, action):
        """Simulate a world transition.

        Args:
          state (object): Current world state to start from.
          action (object): Action to take in current world state.
        None action is not allowed.
        
        Returns:
          (object): Next state.
          (float): Reward or final state value.
          (bool): Is terminal.
        
          or

          If action is not allowed, returns (None, None, None).
        """
        pass


class Modeler(metaclass=ABCMeta):
    """Modeler interface to world model learning logic."""

    @abstractmethod
    def get_dynamics():
        """Passes world dynamics used to simulate transitions.
        
        Return:
          (Dynamics): World dynamics model.
        """
        pass

    @abstractmethod
    def report_step(self, transition):
        """Receives step report from RL loop.
        
        Args:
          transition (Transition): Last transition (state, action, reward, next state, is terminal)
        in RL environment.
        """
        pass

