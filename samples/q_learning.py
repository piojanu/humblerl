import argparse
import humblerl as hrl
from humblerl import Callback, Mind
import numpy as np


class TabularQLearning(Mind, Callback):
    def __init__(self, nstates, nactions, learning_rate, decay_steps, discount_factor):
        # Store training parameters
        self._lr = learning_rate
        self._decay = decay_steps
        self._gamma = discount_factor
        self._episode_count = 1
        self._return = 0
        self.running_avg = 0

        # Initialize Q-table
        self.Q = np.zeros((nstates, nactions), dtype=np.float)

    def plan(self, state, train_mode, debug_mode):
        # Decaying over time random noise for exploration
        random_noise = np.random.randn(self.Q.shape[1]) * (1. / self._episode_count)
        return self.Q[state] + random_noise

    def on_episode_start(self, episode, train_mode):
        self._return = 0

    def on_step_taken(self, step, transition, info):
        # Add reward to return
        self._return += transition.reward

        # Exponentially decaying learning rate
        LR = pow(self._lr, self._episode_count / self._decay)

        # Update Q-table
        if transition.is_terminal:
            target = transition.reward
        else:
            target = transition.reward + self._gamma * \
                np.max(self.Q[transition.next_state])
        self.Q[transition.state, transition.action] += \
            LR * (target - self.Q[transition.state, transition.action])

        # Count episodes
        if transition.is_terminal:
            self._episode_count += 1

    def on_episode_end(self, episode, train_mode):
        self.running_avg = 0.01 * self._return + 0.99 * self.running_avg

    @property
    def metrics(self):
        return {"avg. return": self.running_avg}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HumbleRL tabular Q-Learning sample')
    parser.add_argument('--episodes', type=int, default=865, metavar='N',
                        help='number of episodes to train (default: 865)')
    parser.add_argument('--lr', type=float, default=0.75, metavar='LR',
                        help='learning rate (default: 0.75)')
    parser.add_argument('--decay', type=int, default=400, metavar='N',
                        help='exploration decay steps (default: 400)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                        help='discount factor (default: 0.95)')
    args = parser.parse_args()

    # Create environment and q-learning agent
    env = hrl.create_gym("FrozenLake-v0")
    mind = TabularQLearning(env.state_space, env.action_space.num,
                            learning_rate=args.lr,
                            decay_steps=args.decay,
                            discount_factor=args.gamma)

    # Seed env and numpy
    np.random.seed(7)
    env.env.seed(7)

    # Run training
    hrl.loop(env, mind, n_episodes=args.episodes, callbacks=[mind])
