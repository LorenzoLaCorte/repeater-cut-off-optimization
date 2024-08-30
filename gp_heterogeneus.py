from argparse import ArgumentParser
from ast import Tuple
from collections import defaultdict
import logging
from typing import List

import numpy as np
from gp_utils import OptimizerType, SimParameters, SpaceType, ThresholdExceededError, get_asym_protocol_space, get_t_trunc, optimizerType
from repeater_algorithm import RepeaterChainSimulation
from utility_functions import pmf_to_cdf, secret_key_rate

logging.basicConfig(level=logging.INFO)
cdf_threshold = 0.99

def asym_protocol_runner(simulator, parameters, nodes, idx, space_len):
    """
        The function tests the performance of a single asymmetric protocol,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    protocol = parameters["protocol"]
    dists = sum(1 for item in protocol if 'd' in item)
    
    parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["t_coh"],
                                        nested_swaps=np.log2(nodes+1), nested_dists=np.log2(dists+1))
    
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

    unique_counts = defaultdict(int)

    if store_results:
        with open(f'{filename}', 'w') as file:
            file.write("{\n")
            for skr, protocol in ordered_results:
                unique_counts[skr] += 1
                comma = ',' if protocol != ordered_results[-1][1] else ''
                file.write(f'  "{protocol}": {skr}{comma}\n')
            file.write("}\n\n")
            
    best_results = (ordered_results[0][0], ordered_results[0][1])
    logging.info(f"\nBest results: {best_results}")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--nodes", type=int, default=5, help="Maximum number of levels of SWAP to be performed")
    parser.add_argument("--max_dists", type=int, default=5, help="Maximum round of distillations to be performed")

    parser.add_argument("--optimizer", type=optimizerType, default="gp", help="Optimizer to be used {gp, bf}")
    
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

    parser.add_argument("--t_coh", type=int, default=40000, help="Coherence time")
    parser.add_argument("--p_gen", type=float, default=0.02, help="Generation success probability")
    parser.add_argument("--p_swap", type=float, default=0.85, help="Swapping probability")
    parser.add_argument("--w0", type=float, default=0.99, help="Werner parameter")

    args = parser.parse_args()

    nodes: int = args.nodes
    max_dists: int = args.max_dists

    optimizer: OptimizerType = args.optimizer
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

    simulator = RepeaterChainSimulation()
    brute_force_optimization(simulator, parameters, nodes, max_dists, filename)