import argparse
import cma
import humblerl as hrl
import logging as log
import numpy as np
import os.path
import pickle

from humblerl import Callback, Mind
from multiprocessing import Pool
from tqdm import tqdm


def compute_ranks(x):
    """Computes fitness ranks in rage: [0, len(x))."""
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """Computes ranks and normalize them by the number of samples.
       Finally scale them to the range [−0.5,0.5]"""
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """Agent using CMA-ES algorithm."""

    def __init__(self, n_params, sigma_init=0.1, popsize=100, weight_decay=0.01):
        """Initialize CMA-ES agent.

        Args:
            n_params (int)       : Number of model parameters (NN weights).
            sigma_init (float)   : Initial standard deviation. (Default: 0.1)
            popsize (int)        : Population size. (Default: 100)
            weight_decay (float) : L2 weight decay rate. (Default: 0.01)
        """

        self.weight_decay = weight_decay
        self.population = None

        self.es = cma.CMAEvolutionStrategy(n_params * [0], sigma_init, {'popsize': popsize})

    def ask(self):
        """Returns a list of parameters for new population."""
        self.population = np.array(self.es.ask())
        return self.population

    def tell(self, returns):
        reward_table = np.array(returns)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.population)
            reward_table -= l2_decay
        # Apply fitness shaping function
        reward_table = compute_centered_ranks(reward_table)
        # Convert minimizer to maximizer.
        self.es.tell(self.population, (-1 * reward_table).tolist())

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def save_ckpt(self, path):
        pickle.dump(self, open(os.path.abspath(path), 'wb'))

    @staticmethod
    def load_ckpt(path):
        return pickle.load(open(os.path.abspath(path), 'rb'))


class Liniear(Mind):
    """Simple Artificial Neural Net agent."""

    def __init__(self, input_dim, output_dim):
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.weights = np.zeros((self.in_dim + 1, self.out_dim))

    def plan(self, state, player, train_mode, debug_mode):
        return np.concatenate((state, [1.])) @ self.weights

    def set_weights(self, weights):
        self.weights[:] = weights.reshape(self.in_dim + 1, self.out_dim)

    @property
    def n_weights(self):
        return (self.in_dim + 1) * self.out_dim


class ReturnTracker(Callback):
    """Tracks return."""

    def on_episode_start(self, episode, train_mode):
        self.ret = 0

    def on_step_taken(self, step, transition, info):
        self.ret += transition.reward

    @property
    def metrics(self):
        return {"return": self.ret}


def objective(weights):
    env = hrl.create_gym("CartPole-v0")

    mind = Liniear(env.state_space.shape[0], len(env.valid_actions))
    mind.set_weights(weights)

    history = hrl.loop(env, mind, verbose=0, callbacks=[ReturnTracker()])
    return history['return'][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--popsize', type=int, default=100, metavar='N',
                        help='population size (default: 100)')
    parser.add_argument('--processes', type=int, default=None, metavar='N',
                        help='size of process pool for evaluation (default: CPU count')
    parser.add_argument('--decay', type=float, default=0.01, metavar='L2',
                        help='L2 weight decay rate (default: 0.01)')
    parser.add_argument('--ckpt', type=str, default=None, metavar='PATH',
                        help='checkpoint path to load from/save to model (default: None)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='enable visual play after each epoch (default: False)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug logging (default: False)')
    args = parser.parse_args()

    # Configure logger
    log.basicConfig(level=log.DEBUG if args.debug else log.WARNING,
                    format="[%(levelname)s]: %(message)s")

    # Book keeping variables
    best_return = float('-inf')

    # Create environment and mind
    env = hrl.create_gym("CartPole-v0")
    mind = Liniear(env.state_space.shape[0], len(env.valid_actions))

    # Load CMA-ES solver if ckpt available
    if args.ckpt and os.path.isfile(args.ckpt):
        solver = CMAES.load_ckpt(args.ckpt)
        log.info("Loaded solver from ckpt (NOTE: pop. size and l2 decay was also loaded).")
    else:
        solver = CMAES(mind.n_weights, popsize=args.popsize, weight_decay=args.decay)
        log.info("Created solver with pop. size: %d and l2 decay: %f.", args.popsize, args.decay)

    # Train for N epochs
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        # Get new population
        population = solver.ask()

        with Pool(processes=args.processes) as pool:
            returns = pool.map(objective, population)

        pbar.set_postfix(best=best_return, current=max(returns))
        best_return = max(best_return, max(returns))

        # Update solver
        solver.tell(returns)

        if args.ckpt:
            solver.save_ckpt(args.ckpt)
            log.debug("Saved checkpoint in path: %s", args.ckpt)

        if args.render:
            # Evaluate current parameters with render
            mind.set_weights(solver.current_param())
            history = hrl.loop(env, mind, render_mode=True, verbose=0, callbacks=[ReturnTracker()])
            log.info("Current parameters (weights) return: %f.", history['return'][0])

    # If environment wasn't solved then exit with error
    assert best_return == 200, "Environment wasn't solved!"
