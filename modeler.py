from abc import ABCMeta, abstractmethod

class Dynamics(metaclass=ABCMeta):
    """Dynamics interface to world model."""

    @abstractmethod
    def __call__(self, state, action):
        """Simulate a world transition.

        Args:
          state (object): Current world state to start from.
          action (object): Action to take in current world state.
        
        Returns:
          (object): Next state after taking the given action in the given state.
        """
        pass

class Modeler(metaclass=ABCMeta):
    """Modeler interface to world model learner."""

    @abstractmethod
    def get_dynamics():
        """Passes world dynamics used to simulate transitions.
        
        Return:
          (Dynamics): World dynamics.
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

