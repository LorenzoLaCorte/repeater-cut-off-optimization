"""
This script is used to test the performance of different distillation strategies
 in an extensive way, by using a Gaussian Process to optimize the number of distillations.
"""

from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple, Optional, Union

import logging
import numpy as np

from skopt import gp_minimize
from skopt.space import Real
from scipy.optimize import OptimizeResult
from skopt.utils import use_named_args

from gp_plots import plot_optimization_process
from gp_utils import (
    optimizerType, OptimizerType, ThresholdExceededError, SimParameters, # Typing
    get_asym_protocol_space,  # Getters for Spaces
    get_protocol_from_center_spacing_symmetricity, # Getters for Protocols
    get_t_trunc, get_ordered_results, # Other Getters
) 

from repeater_algorithm import RepeaterChainSimulation
from utility_functions import pmf_to_cdf, secret_key_rate

logging.basicConfig(level=logging.INFO)
cdf_threshold = 0.99


def write_results(filename, ordered_results):
    """
    TODO: Write the docstring
    TODO: refactor to some utils and check if its necessary the one there
    """
    with open(f'{filename}', 'w') as file:
        file.write("{\n")
        for skr, protocol in ordered_results:
            comma = ',' if protocol != ordered_results[-1][1] else ''
            file.write(f'  "{protocol}": {skr}{comma}\n')
        file.write("}\n\n")
        file.write(f"Unique values: {len(set([skr for skr, _ in ordered_results]))}\n")


def asym_protocol_runner(simulator, parameters, nodes, idx=None, space_len=None):
    """
        The function tests the performance of a single asymmetric protocol,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    protocol = parameters["protocol"]
    dists = sum(1 for item in protocol if 'd' in item)
    
    if isinstance(parameters["p_gen"], list):
        t_coh = parameters["t_coh"] * 299792458 / max(L0)
        parameters["t_trunc"] = get_t_trunc(min(parameters["p_gen"]), parameters["p_swap"], t_coh,
                                            nested_swaps=np.log2(nodes+1), nested_dists=np.log2(dists+1))
    else:        
        parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["t_coh"],
                                        nested_swaps=np.log2(nodes+1), nested_dists=np.log2(dists+1))
    
    if (idx is None) or (space_len is None):
        logging.info(f"\nRunning: {parameters}")
    else:
        logging.info(f"\n({idx+1}/{space_len}) Running: {parameters}")
    
    pmf, w_func = simulator.asymmetric_protocol(parameters, nodes-1)

    skr = secret_key_rate(pmf, w_func, parameters["t_trunc"])
    logging.info(f"Protocol {parameters['protocol']},\t r = {skr}\n")
    return skr, pmf, w_func


def brute_force_optimization(simulator, parameters: SimParameters, 
                             nodes: int, max_dists: int,
                             filename: str, store_results: bool = True) -> None:
    """
    This function is used to test the performance of all possible heterogeneus protocols
    given: 
        - a number N of nodes and S=(N-1) of segments
        - a number k of maximum distillation per segment per level of swapping
    It bruteforces all the possible configurations and finds the maximum rate achieved and the best protocol yielding it.
    """
    results: List[Tuple[np.float64, Tuple[str]]] = []

    space = get_asym_protocol_space(nodes, max_dists)
    logging.info(f"The space of asymmetric protocols has size {len(space)}")

    for idx, protocol in enumerate(space):
        try:
            parameters["protocol"] = protocol
            secret_key_rate, pmf, _ = asym_protocol_runner(simulator, parameters, nodes, idx, len(space))
            
            cdf_coverage = pmf_to_cdf(pmf)[-1]
            if cdf_coverage < cdf_threshold:
                raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})
            
            results.append((secret_key_rate, protocol))
        
        except ThresholdExceededError as e:
            logging.warn((f"Simulation under coverage for {protocol}"
                            "\nSet a (higher) fixed truncation time (--t_trunc)"))

    ordered_results: List[Tuple[np.float64, Tuple[str]]] = sorted(results, key=lambda x: x[0], reverse=True)

    if store_results:
        write_results(filename, ordered_results)
            
    best_results = (ordered_results[0][0], ordered_results[0][1])
    logging.info(f"\nBest results: {best_results}")


# Cache results of the objective function to avoid re-evaluating the same point and speed up the optimization
# TODO: maybe I can simply decorate the function with lru_cache
cache_results = defaultdict(np.float64)
strategy_to_protocol = {}

def objective_key_rate(space, nodes, max_dists, shot_count, gp_shots, parameters, simulator):
    """
    Objective function, consider the whole space of actions,
        returning a negative secret key rate
        in order for the optimizer to maximize the function.
    """
    shot_count[0] += 1
    gamma = space['gamma']
    zeta = space['zeta']
    
    logging.info(f"\n\nGenerating a protocol with gamma={gamma}, zeta={zeta}...")
    parameters["protocol"] = get_protocol_from_center_spacing_symmetricity(gamma, zeta, nodes, max_dists)
    logging.info(f"Protocol generated: {parameters['protocol']}")
    strategy_to_protocol[(gamma, zeta)] = parameters["protocol"]

    if parameters["protocol"] in cache_results:
        logging.info("Already evaluated protocol, returning cached result")
        return -cache_results[parameters["protocol"]]
    
    secret_key_rate, pmf, _ = asym_protocol_runner(simulator, parameters, nodes, shot_count[0], gp_shots)
    
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
    # Add protocol to result dict, to then retrieve it later
    result.x_iters[-1].append(strategy_to_protocol[(result.x_iters[-1][0], result.x_iters[-1][1])])
    

def gaussian_optimization(simulator, parameters: SimParameters, 
                          nodes: int, max_dists: int, 
                          gp_shots: Optional[int], gp_initial_points: Optional[int],
                          filename: str, store_results: bool = True) -> None:
    """
    This function is used to test the performance of different protocols in an extensive way.
    """
    logging.info(f"\n\nNumber of nodes: {nodes}, max dists: {max_dists}")
    logging.info(f"Gaussian process with {gp_shots} evaluations "
                f"and {gp_initial_points} initial points\n\n")

    shot_count = [-1]

    space = [
        Real(0, 1, name='gamma'),
        Real(0, 1, name='zeta'), 
    ]
    @use_named_args(space)
    def wrapped_objective(**space_params):
        return objective_key_rate(space_params, nodes, max_dists, shot_count, gp_shots, parameters, simulator)

    try:
        result: OptimizeResult = gp_minimize(
            wrapped_objective,
            space,
            n_calls=gp_shots,
            n_initial_points=gp_initial_points,
            callback=[is_gp_done],
            acq_func='LCB',
            kappa=1.96,
            noise=1e-10,    # There is no noise in results
        )
        
        ordered_results: List[Tuple[np.float64, Tuple[int]]] = get_ordered_results(
            result=result, space_type="asymmetric", number_of_swaps=None)

        if store_results:
            plot_optimization_process(min_dists=0, max_dists=max_dists, parameters=parameters, 
                                      results=ordered_results, gp_result=result)
        cache_results.clear()
    
    except ThresholdExceededError as e:
        logging.warn((f"Simulation under coverage for {parameters['protocol']}"))
    
    if store_results:
        write_results(filename, ordered_results)
            
    best_results = (ordered_results[0][0], ordered_results[0][1])
    logging.info(f"\nBest results: {best_results}")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--nodes", type=int, default=4, help="Number of nodes in the chain")
    parser.add_argument("--max_dists", type=int, default=1, help="Maximum round of distillations of each segment per level")

    parser.add_argument("--optimizer", type=optimizerType, default="bf", help="Optimizer to be used {gp, bf}")
    
    parser.add_argument("--gp_shots", type=int, default=100,
                        help=(  "Number of shots for Gaussian Process optimization"
                                "If not specified, it is computed dynamically based on the protocol"))
    
    parser.add_argument("--gp_initial_points", type=int, default=10,
                        help=(  "Number of initial points for Gaussian Process optimization"
                                "If not specified, it is computed dynamically based on the protocol"))
       
    parser.add_argument("--filename", type=str, default='output.txt', help="Filename for output log")
    parser.add_argument("--cdf_threshold", type=float, default=0.99, 
                        help=("Threshold for CDF coverage. If one configuration goes below this threshold, "
                              "the simulation is discarded"))
    
    parser.add_argument("--t_coh", type=float, default=20, help="Coherence time")
    parser.add_argument("--p_swap", type=float, default=0.85, help="Swapping probability")
    parser.add_argument("--p_gen", type=float, nargs='+', default=[0.002588,0.0009187,0.0009082], help="Generation success probability")
    parser.add_argument("--w0", type=float, nargs='+', default=[0.9577,0.9524,0.9523], help="Werner parameter")
    parser.add_argument("--L0", type=int, nargs='+', default=[19800,50400,60400], help="Length of links")
    
    args = parser.parse_args()

    nodes: int = args.nodes
    max_dists: int = args.max_dists

    optimizer: OptimizerType = args.optimizer
    gp_shots: int = args.gp_shots
    gp_initial_points: int = args.gp_initial_points

    filename: str = args.filename
    cdf_threshold: float = args.cdf_threshold

    t_coh: Union[int, List[int]] = args.t_coh
    p_swap: Union[float, List[float]] = args.p_swap
    p_gen: Union[float, List[float]] = args.p_gen if len(args.p_gen) > 1 else args.p_gen[0]
    w0: Union[float, List[float]] = args.w0 if len(args.w0) > 1 else args.w0[0]

    # Ensure all are lists if one is a list
    if isinstance(p_gen, list):
        if not all(isinstance(arg, list) for arg in [args.p_gen, args.w0, args.L0]) or \
            not all(len(arg) == len(p_gen) for arg in [args.p_gen, args.w0, args.L0]):
            raise ValueError("Parameters must be all lists or all scalars")
        L0: List[int] = args.L0

    parameters: SimParameters = {
        't_coh': t_coh,
        'p_gen': p_gen,
        'p_swap': p_swap,
        'w0': w0,
    }

    if isinstance(p_gen, list):
        parameters['L0'] = L0
        
    simulator = RepeaterChainSimulation()
    
    # Start the optimization process
    if optimizer == "gp":
        gaussian_optimization(simulator, parameters, 
                              nodes, max_dists, 
                              gp_shots, gp_initial_points, 
                              filename)
    elif optimizer == "bf":
        brute_force_optimization(simulator, parameters, 
                                 nodes, max_dists, 
                                 filename)