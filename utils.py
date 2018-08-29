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
