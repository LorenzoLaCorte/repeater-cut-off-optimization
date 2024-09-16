"""
This script is used to test the performance of different distillation strategies
 in an extensive way, by using a Gaussian Process to optimize the number of distillations.
"""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import List, Tuple, Optional

import logging

import matplotlib.pyplot as plt
colorblind_palette = [
    "#0072B2",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#000000",
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)

import numpy as np

from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from scipy.optimize import OptimizeResult
from skopt.utils import use_named_args
from sklearn.gaussian_process import GaussianProcessRegressor

from gp_plots import plot_optimization_process
from gp_utils import (
    write_results, # Utils
    get_sym_protocol_space, get_sym_protocol_space_size,  # Getters for Spaces
    get_protocol_from_distillations, get_protocol_from_strategy, get_protocol_from_center_and_spacing, # Getters for Protocols
    get_all_maxima, get_t_trunc, get_ordered_results, # Other Getters
) 

from repeater_types import SpaceType, optimizerType, OptimizerType, ThresholdExceededError, SimParameters, spaceType # Typing

from repeater_algorithm import RepeaterChainSimulation
from utility_functions import secret_key_rate, pmf_to_cdf
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data
)
logging.getLogger().level = logging.INFO

fixed_t_trunc = None
cdf_threshold = 0.99


def sim_distillation_strategies(simulator, parameters):
    """
        Fixed parameters:
            - number of swaps
            - hardware paramters

        The function tests the performance of different distillation strategies,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    if fixed_t_trunc is not None:
        parameters["t_trunc"] = fixed_t_trunc
    else:
        parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["t_coh"],
                                            parameters["protocol"].count(0), parameters["protocol"].count(1))

    logging.info(f"\nRunning: {parameters}")
    pmf, w_func = simulator.nested_protocol(parameters)

    rate = secret_key_rate(pmf, w_func, parameters["t_trunc"])
    logging.info(f"Protocol {parameters['protocol']},\t r = {rate}\n")
    return rate, pmf, w_func


def brute_force_optimization(simulator, parameters: SimParameters, space_type: SpaceType, 
                             min_swaps: int, max_swaps: int, min_dists: int, max_dists: int, 
                             filename: str, store_results: bool = True) -> None:
    """
    This function is used to test the performance of different distillation strategies
    by bruteforcing all the possible configurations and returning the maximum rate achieved.
    """
    if store_results:
        with open(filename, 'w') as file:
            file.write(f"From {min_swaps} to {max_swaps} swaps\nFrom {min_dists} to {max_dists} distillations\n")
            file.write(f"Bruteforce process for all the possible evaluations\n\n")

    best_results = {}

    for number_of_swaps in range(min_swaps, max_swaps+1):
        logging.info(f"\n\nNumber of swaps: {number_of_swaps}")
        results: List[Tuple[np.float64, Tuple[int]]] = []

        space = []
        if space_type == "one_level":
            if min_dists == 0:
                space.append(tuple([0]*number_of_swaps))
            for number_of_dists in range(max(1, min_dists), max_dists+1):
                for where_to_distill in range(number_of_swaps+1):
                    space.append(get_protocol_from_distillations(number_of_swaps, number_of_dists, where_to_distill))
        elif space_type == "enumerate":
            space = get_sym_protocol_space(min_dists, max_dists, number_of_swaps)
        
        protocol_space_size = get_sym_protocol_space_size(space_type, min_dists, max_dists, number_of_swaps)
        assert len(space) == protocol_space_size, \
            f"Expected {protocol_space_size} protocols, got {len(space)}"

        for protocol in space:
            try:
                parameters["protocol"] = protocol
                secret_key_rate, pmf, _ = sim_distillation_strategies(simulator, parameters)
                
                cdf_coverage = pmf_to_cdf(pmf)[-1]
                if cdf_coverage < cdf_threshold:
                    raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})
                
                results.append((secret_key_rate, protocol))
            
            except ThresholdExceededError as e:
                logging.warn((f"Simulation under coverage for {number_of_swaps}"
                              "\nSet a (higher) fixed truncation time (--t_trunc)"))
                best_results[number_of_swaps] = (None, None, e.extra_info['cdf_coverage'])
                break

        if store_results:
            plot_optimization_process(min_dists, max_dists, parameters, results=results, number_of_swaps=number_of_swaps)

        ordered_results: List[Tuple[np.float64, Tuple[int]]] = sorted(results, key=lambda x: x[0], reverse=True)
        best_results[number_of_swaps] = (ordered_results[0][0], ordered_results[0][1], None)

    if store_results:
        write_results(filename, parameters, best_results)
    return best_results

# Cache results of the objective function to avoid re-evaluating the same point and speed up the optimization
cache_results = defaultdict(np.float64)
strategy_to_protocol = {}

def objective_key_rate(space, space_type: SpaceType, number_of_swaps, parameters, simulator):
    """
    Objective function, consider the whole space of actions,
        returning a negative secret key rate
        in order for the optimizer to maximize the function.
    """
    if space_type == "one_level":
        number_of_dists = space['rounds of distillation']
        where_to_distill = space['after how many swaps we distill']
        parameters["protocol"] = get_protocol_from_distillations(
                                    number_of_swaps=number_of_swaps, 
                                    number_of_dists=number_of_dists, 
                                    where_to_distill=where_to_distill)
    
    elif space_type == "strategy":
        number_of_dists = space['rounds of distillation']
        strategy = space['strategy']
        strategy_weight = space['strategy weight']
        parameters["protocol"] = get_protocol_from_strategy(strategy, strategy_weight, number_of_swaps, number_of_dists)
        strategy_to_protocol[(number_of_dists, strategy, strategy_weight)] = parameters["protocol"]

    elif space_type == "centerspace":
        number_of_dists = space['rounds of distillation']
        eta = space['eta']
        tau = space['tau']
        logging.debug(f"\n\nGenerating a protocol with k={number_of_dists}, eta={eta}, tau={tau}...")
        parameters["protocol"] = get_protocol_from_center_and_spacing(eta, tau, number_of_swaps, number_of_dists)
        strategy_to_protocol[(number_of_dists, eta, tau)] = parameters["protocol"]

    if parameters["protocol"] in cache_results:
        logging.debug("Already evaluated protocol, returning cached result")
        return -cache_results[parameters["protocol"]]
    
    secret_key_rate, pmf, _ = sim_distillation_strategies(simulator, parameters)
    
    cdf_coverage = pmf_to_cdf(pmf)[-1]
    if cdf_coverage < cdf_threshold:
        raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})                

    cache_results[parameters["protocol"]] = secret_key_rate
    space.update({'protocol': parameters["protocol"]})
    # The gaussian process minimizes the function, so return the negative of the key rate 
    return -secret_key_rate


def is_gp_done(result: OptimizeResult):
    """
    Callback function to stop the optimization if all the points have been evaluated.
    """
    # TODO: maybe, I can result.space.bounds to substitute protocol_space_size
    # protocols_evaluated = len(list(set([ast.literal_eval(r[0]) for r in result.x_iters])))
    # assert protocols_evaluated == len(cache_results), f"Expected {protocols_evaluated} protocols, got {len(cache_results)}"
    
    # Add protocol to result dict, to then retrieve it later
    # TODO: the check can be done in some other way (type checking), in order to be more robust
    if len(result.x_iters[-1]) == 3:
        result.x_iters[-1].append(strategy_to_protocol[(result.x_iters[-1][0], result.x_iters[-1][1], result.x_iters[-1][2])])
    
    if len(cache_results) == protocol_space_size:
        logging.info(f"All protocols evaluated ({len(cache_results)}/{protocol_space_size}), stopping optimization")
        
        if len(result.models) == 0:
            logging.info(f"Evaluation terminated within initial points: protocol space is too small or initial points are too many")
            result.models = [GaussianProcessRegressor] # Use a random model for the partial dependence plot
        return True
    

def gaussian_optimization(simulator, parameters: SimParameters, space_type: SpaceType, 
                          min_swaps: int, max_swaps: int, min_dists: int, max_dists: int, 
                          gp_shots: Optional[int], gp_initial_points: Optional[int],
                          filename: str, store_results: bool = True) -> None:
    """
    This function is used to test the performance of different protocols in an extensive way.
    """
    if store_results:
        with open(filename, 'w') as file:
            file.write(f"From {min_swaps} to {max_swaps} swaps\nFrom {min_dists} to {max_dists} distillations\n")

    best_results = {}

    for number_of_swaps in range(min_swaps, max_swaps+1):
        logging.info(f"\n\nNumber of swaps: {number_of_swaps}")
        
        global protocol_space_size # TODO: refactor in order to avoid using global variables
        protocol_space_size = get_sym_protocol_space_size(space_type, min_dists, max_dists, number_of_swaps)

        # Define reasonable default values (in terms of a percentage of the protocol space size)
        if gp_initial_points is None:
            gp_initial_points_def = 10 + int(protocol_space_size * .10)
        if gp_shots is None:
            gp_shots_def = (gp_initial_points or gp_initial_points_def) + 1 + int(protocol_space_size * .20)

        if store_results:
            with open(filename, 'w') as file:
                file.write((f"Gaussian process with {gp_shots or gp_shots_def} evaluations "
                            f"and {gp_initial_points or gp_initial_points_def} initial points\n\n"))

        if space_type == "one_level":
            space = [
                Integer(min_dists, max_dists, name='rounds of distillation'), 
                Integer(0, number_of_swaps, name='after how many swaps we distill'),
            ]
            @use_named_args(space)
            def wrapped_objective(**space_params):
                return objective_key_rate(space_params, space_type, number_of_swaps, parameters, simulator)
        
        elif space_type == "strategy":
            space = [
                Integer(min_dists, max_dists, name='rounds of distillation'), 
                Categorical(["dists_first", "swaps_first", "alternate"], name='strategy'),
                Real(0.5, 1.0, name='strategy weight'),
            ]
            @use_named_args(space)
            def wrapped_objective(**space_params):
                return objective_key_rate(space_params, space_type, number_of_swaps, parameters, simulator)
            
        elif space_type == "centerspace":
            space = [
                Integer(min_dists, max_dists, name='rounds of distillation'), 
                Real(-1, 1, name='eta'), 
                Real(0.099, 1, name='tau'), # TODO: if tau < 0.1, PDF values are sometimes null and protocols are invalid
            ]
            @use_named_args(space)
            def wrapped_objective(**space_params):
                return objective_key_rate(space_params, space_type, number_of_swaps, parameters, simulator)

        try:
            # TODO: if min_dist > 0, I clearly wanna skip this
            # Give NO-DIST (only swaps) protocol as initial point
            if space_type == "one_level":
                x0 = [0, 0]
            elif space_type == "strategy":
                x0 = [0, "dists_first", 1.0]
            elif space_type == "centerspace":
                x0 = [0, -1, 0.1]

            # Perform the optimization
            result: OptimizeResult = gp_minimize(
                wrapped_objective,
                space,
                n_calls=(gp_shots or gp_shots_def),
                n_initial_points=(gp_initial_points or gp_initial_points_def),
                callback=[is_gp_done],
                x0=x0,
                acq_func='LCB',
                kappa=1.96 * 2, # Double the default kappa, to prefer exploration
                noise=1e-10,    # There is no noise in results
            )
            
            ordered_results: List[Tuple[np.float64, Tuple[int]]] = get_ordered_results(result, space_type, number_of_swaps)

            if store_results:
                plot_optimization_process(min_dists, max_dists, parameters, 
                                          results=ordered_results, gp_result=result,
                                          number_of_swaps=number_of_swaps)
            
            cache_results.clear()

        except ThresholdExceededError as e:
            best_results[number_of_swaps] = (None, None, e.extra_info['cdf_coverage'])
            continue          
    
        # Get the best parameters and score from results
        best_results[number_of_swaps] = (ordered_results[0][0], ordered_results[0][1], None)
    
    if store_results:
        write_results(filename, parameters, best_results)
    return best_results


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=2, help="Minimum number of levels of SWAP to be performed")
    parser.add_argument("--max_swaps", type=int, default=2, help="Maximum number of levels of SWAP to be performed")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum round of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=2, help="Maximum round of distillations to be performed")
    
    parser.add_argument("--optimizer", type=optimizerType, default="gp", help="Optimizer to be used {gp, bf}")
    
    parser.add_argument("--space", type=spaceType, default="centerspace", 
                        help="Space to be tested {one_level, enumerate, strategy, centerspace}")
    
    parser.add_argument("--gp_shots", type=int,
                        help=(  "Number of shots for Gaussian Process optimization"
                                "If not specified, it is computed dynamically based on the protocol"))
    
    parser.add_argument("--gp_initial_points", type=int, 
                        help=(  "Number of initial points for Gaussian Process optimization"
                                "If not specified, it is computed dynamically based on the protocol"))
                        
    parser.add_argument("--filename", type=str, default='output.txt', help="Filename for output log")
    parser.add_argument("--cdf_threshold", type=float, default=0.99, 
                        help=("Threshold for CDF coverage. If one configuration goes below this threshold, "
                              "the simulation is discarded"))

    parser.add_argument("--t_coh", type=int, default=1440000, help="Coherence time")
    parser.add_argument("--p_gen", type=float, default=0.0026, help="Generation success probability")
    parser.add_argument("--p_swap", type=float, default=0.85, help="Swapping probability")
    parser.add_argument("--w0", type=float, default=0.9577, help="Werner parameter")

    parser.add_argument("--dp", action='store_true', default=False, 
                        help="Use dynamic programming to cache results and set a fixed truncation time")
    parser.add_argument("--t_trunc", type=int, 
                        help=(  "Fixed truncation time. In case of dynamic programming, it is fixed to this value "
                                "or a default value is computed. "
                                "In case of no dynamic programming, it is computed dynamically for each simulation."))
    
    args: Namespace = parser.parse_args()

    min_swaps: int = args.min_swaps
    max_swaps: int = args.max_swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists

    optimizer: OptimizerType = args.optimizer
    space: SpaceType = args.space
    gp_shots: int = args.gp_shots
    gp_initial_points: int = args.gp_initial_points

    filename: str = args.filename
    cdf_threshold: float = args.cdf_threshold

    t_coh = args.t_coh
    p_gen = args.p_gen
    p_swap = args.p_swap
    w0 = args.w0

    parameters: SimParameters = {
        't_coh': t_coh,
        'p_gen': p_gen,
        'p_swap': p_swap,
        'w0': w0,
    }
    fixed_t_trunc = args.t_trunc

    dp_enabled = args.dp
    simulator = None

    # Set up the caching of results for dp
    if dp_enabled:
        simulator = RepeaterChainSimulation(use_cache=True)
        # In case of dynamic programming, a fixed truncation time is required
        if args.t_trunc is None:
            fixed_t_trunc = get_t_trunc(p_gen, p_swap, t_coh, max_swaps, max_dists)
    else:
        simulator = RepeaterChainSimulation(use_cache=False)

    # Abort in case of bad combinations of optimizer and space
    if optimizer == "bf" and space == "strategy":
        raise ValueError("Bruteforce not supported for strategy space, use 'enumerate'")
    elif optimizer == "gp" and space == "enumerate":
        raise ValueError("Gaussian Process not supported for enumerate space, use 'one_level' or 'strategy'")
    
    # If the gp shots are enough to bruteforce the biggest (max. swap) solution, use a bf algorithm
    if gp_shots is not None and gp_shots >= get_sym_protocol_space_size(space, min_dists, max_dists, max_swaps):
        logging.info("GP shots are enough to bruteforce the solutions, using brute force algorithm")
        optimizer = "bf"
        gp_shots = None
    
    # Start the optimization process
    if optimizer == "gp":
        gaussian_optimization(simulator, parameters, space, 
                              min_swaps, max_swaps, min_dists, max_dists, 
                              gp_shots, gp_initial_points, filename)
    elif optimizer == "bf":
        brute_force_optimization(simulator, parameters, space, 
                                 min_swaps, max_swaps, min_dists, max_dists, 
                                 filename)