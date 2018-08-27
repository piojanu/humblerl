import argparse
import cma
import humblerl as hrl
import logging as log
import numpy as np
import os.path
import pickle

from functools import partial
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


class NN(Mind):
    """Simple Artificial Neural Net agent."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.out_dim = output_dim

        self.h = np.zeros((self.in_dim, self.h_dim))
        self.h_bias = np.zeros(self.h_dim)
        self.out = np.zeros((self.h_dim, self.out_dim))
        self.out_bias = np.zeros(self.out_dim)

    def plan(self, state, player, train_mode, debug_mode):
        h = state @ self.h + self.h_bias
        h[h < 0] = 0

        out = h @ self.out + self.out_bias
        return out, None

    def set_weights(self, weights):
        h_offset = self.in_dim * self.h_dim
        h_bias_offset = h_offset + self.h_dim
        out_offset = h_bias_offset + self.h_dim * self.out_dim

        self.h[:] = weights[:h_offset].reshape(self.h.shape)
        self.h_bias[:] = weights[h_offset:h_bias_offset]
        self.out[:] = weights[h_bias_offset:out_offset].reshape(self.out.shape)
        self.out_bias[:] = weights[out_offset:]


    @property
    def n_weights(self):
        return self.h_dim * (self.in_dim + self.out_dim) + self.h_dim + self.out_dim


class ReturnTracker(Callback):
    """Tracks return."""

    def on_loop_start(self):
        self.ret = 0

    def on_step_taken(self, transition, info):
        self.ret += transition.reward

    @property
    def value(self):
        return self.ret

def objective(weights, game_name, nn_dims):
    env = hrl.create_gym(game_name)
    
    nn = NN(*nn_dims)
    nn.set_weights(weights)

    tracker = ReturnTracker()
    
    hrl.loop(env, nn, verbose=0, callbacks=[tracker])
    return tracker.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--game_name', type=str, default="CartPole-v0", metavar='S',
                        help='OpenAI Gym, cont. states and disc. actions (default: CartPole-v0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--popsize', type=int, default=100, metavar='N',
                        help='population size (default: 100)')
    parser.add_argument('--processes', type=int, default=None, metavar='N',
                        help='size of process pool for evaluation (default: CPU count')
    parser.add_argument('--decay', type=float, default=0.01, metavar='L2',
                        help='L2 weight decay rate (default: 0.01)')
    parser.add_argument('--h_dim', type=int, default=16, metavar='N',
                        help='size of hidden layer (default: 16)')
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
    
    # Create environment to get state-action space info
    env = hrl.create_gym(args.game_name)
    nn_dims = (env.state_space.shape[0], args.h_dim, len(env.valid_actions))

    # Create neural net to get number of weights
    nn = NN(*nn_dims)
    n_weights = nn.n_weights
    
    # Load CMA-ES solver if ckpt available
    if args.ckpt and os.path.isfile(args.ckpt):
        solver = CMAES.load_ckpt(args.ckpt)
        log.info("Loaded solver from ckpt (NOTE: pop. size and l2 decay was also loaded).")
    else:
        solver = CMAES(n_weights, popsize=args.popsize, weight_decay=args.decay)
        log.info("Created solver with pop. size: %d and l2 decay: %f.", args.popsize, args.decay)

    # Train for N epochs
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        # Get new population
        population = solver.ask()

        with Pool(processes=args.processes) as pool:
            returns = pool.map(
                partial(objective, game_name=args.game_name, nn_dims=nn_dims), population)

        pbar.set_postfix(best=best_return, current=max(returns))
        best_return = max(best_return, max(returns))

        # Update solver
        solver.tell(returns)

        if args.ckpt:
            solver.save_ckpt(args.ckpt)
            log.debug("Saved checkpoint in path: %s", args.ckpt)

        if args.render:
            # Evaluate current parameters with render
            tracker = ReturnTracker()
            nn.set_weights(solver.current_param())
            hrl.loop(env, nn, render_mode=True, verbose=0, callbacks=[tracker])
            log.info("Current parameters (weights) return: %f.", tracker.value)
            
