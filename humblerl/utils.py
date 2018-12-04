from .environments import Environment

class History(object):
    """Keeps history of metrics."""

    def __init__(self):
        self.episode = 0
        self.history = {}

    def update(self, metrics):
        """Updates history dict in such a way, that every metric values list has the same length
        equal number of episodes."""
        for k in set(list(self.history.keys()) + list(metrics.keys())):
            self.history.setdefault(k, [None] * self.episode).append(metrics.get(k, None))

        self.episode += 1


class QualitativeRewardEnvironment(Environment):
    """Wrapper on Environment to achive reward only at the end of game."""

    def __init__(self, env):
        self.env = env

    def render(self):
        return self.env.render()

    def reset(self, train_mode=True):
        return self.env.reset(train_mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state_space(self):
        return self.env.state_space

    @property
    def current_state(self):
        return self.env.current_state

    @property
    def valid_actions(self):
        return self.env.valid_actions

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            if reward > 0:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        return state, reward, done, info


def unpack(value, number=2, default=None):
    """Unpack given `value` (item/tuple/list) to `number` of elements.
    Elements from `value` goes first, then the rest is set to `default`."""

    if not isinstance(value, list):
        if isinstance(value, tuple):
            value = list(value)
        else:
            value = [value]

    assert len(value) <= number

    for _ in range(number - len(value)):
        value.append(default)

    return value
