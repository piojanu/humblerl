from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


class Environment(object):
    """Interface for environments."""

    def reset(self, train_mode=True, context=None):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. (default: True)
            context (Trainer): Training context. E.g. training options like
        learning rate, loggers like TensorBoard etc. (default: None)

        Returns:
            np.array: The initial state. 
        """

        raise NotImplementedError()

    def step(self, action, context=None):
        """Perform action in environment and return new state, reward and done flag.

        Args:
            action (list of floats): Action to perform. In discrete action space
        it's single element list with action number.
            context (Trainer): Training context. E.g. training options like
        learning rate, loggers like TensorBoard etc. (default: None)

        Returns:
            np.array: New state.
            float: Next reward.
            bool: Flag indicating if episode has ended.
        """

        raise NotImplementedError()
