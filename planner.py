from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):
    """Abstract class representing policy in Reinforcement Learning task."""

    @abstractmethod
    def __call__(self, state):
        """Evaluate policy and return action.
        
        Args:
          state (object): Current world state to start from.

        Returns:
            list of floats: action to take in the environment.
        """
        pass


class Planner(metaclass=ABCMeta):
    """Planner interface to model-based policy learning logic."""

    def __init__(self, model):
        """Initialize planner.
        
        Args:
          model (Dynamics): World dynamics model. 
        """

        self._model = model

    @abstractmethod
    def get_policy(self):
        """Passes policy used to select actions.
        
        Return:
          (Policy): Policy.
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
