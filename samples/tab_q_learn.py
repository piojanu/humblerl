from humblerl import Agent, Policy
from humblerl import OpenAIGymWrapper

import gym
import numpy as np
import matplotlib.pyplot as plt


class TabularQLearning(Policy):
    def __init__(self, nstates, nactions, learning_rate=0.9, decay_steps=300, discount_factor=0.95):
        # Store training parameters
        self._lr = learning_rate
        self._decay = decay_steps
        self._gamma = discount_factor
        self._episode_count = 1

        # Initialize Q-table
        self.Q = np.zeros([nstates, nactions], dtype=np.float)

    def select_action(self, curr_state):
        # Decaying over time random noise for exploration
        random_noise = np.random.randn(self.Q.shape[1]) * (1./self._episode_count)
        action = np.argmax(self.Q[curr_state] + random_noise)
        return action

    def report(self, transition):
        # Exponentially decaying learning rate
        LR = pow(self._lr, self._episode_count/self._decay)

        # Update Q-table
        target = transition.reward + self._gamma * np.max(self.Q[transition.next_state])
        self.Q[transition.state, transition.action] += \
            LR * (target - self.Q[transition.state, transition.action])

        # Count episodes
        if transition.is_terminal:
            self._episode_count += 1


if __name__ == "__main__":
    env = OpenAIGymWrapper(gym.make("FrozenLake-v0"))
    model = TabularQLearning(env.state_space_info.size, env.action_space_info.size)
    agent = Agent(env, model)

    # Seed env and numpy for predictability
    np.random.seed(77)
    env._env.seed(77)

    NEPISODES = 1500
    cum_return = 0
    episodes_len = []
    returns = []
    for i, episode in enumerate(range(NEPISODES)):
        agent.reset()

        # Play for one episode, but no longer then 'max_steps'
        for j, transition in enumerate(agent.play(max_steps=100)):
            if transition.is_terminal:
                cum_return += transition.reward
                episodes_len.append(j)

        # Book keeping
        if (i + 1) % 100 == 0:
            print("[{:3.0f}%] Avg. reward in 100 episodes: {}".format(
                i/NEPISODES * 100, cum_return/100))
            returns.append(cum_return)
            cum_return = 0

    print("Final Q function tab:")
    print(model.Q)

    # Plot episodes lengths and returns
    plt.subplot(2, 1, 1)
    plt.plot(returns, label="Returns")
    plt.gca().legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(episodes_len, label="Episodes lengths")
    plt.gca().legend(loc='best')
    plt.show()
