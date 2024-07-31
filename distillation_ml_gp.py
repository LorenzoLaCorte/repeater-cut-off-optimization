"""
This script is used to test the performance of different distillation strategies
 in an extensive way, by using a Gaussian Process to optimize the number of distillations.
"""

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import itertools
import logging
import ast
from typing import List, Tuple

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
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
from matplotlib.ticker import MaxNLocator

import copy
import numpy as np

from skopt import gp_minimize, dummy_minimize
from skopt.space import Integer, Categorical
from scipy.optimize import OptimizeResult
from skopt.utils import use_named_args

from distillation_ml_plots import plot_objective, plot_convergence

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)

from utility_functions import secret_key_rate
from utility_functions import pmf_to_cdf

from scipy.special import binom


def index_lowercase_alphabet(i):
    """
    Takes in input an integer i and returns the corresponding lowercase letter in the alphabet.
    """
    return chr(i + 97)


def remove_unstable_werner(pmf, w_func, threshold=1.0e-15):
    """
    Removes unstable Werner parameters where the probability mass is below a specified threshold
    and returns a new Werner parameter array without modifying the input array.

    Parameters:
    - pmf (np.array): The probability mass function array.
    - w_func (np.array): The input Werner parameter array.
    - threshold (float): The threshold below which Werner parameters are considered unstable.
    
    Returns:
    - np.array: A new Werner parameter array with unstable parameters removed.

     note: it is used in plots to remove oscillations in the plot.
    """
    new_w_func = w_func.copy()
    for t in range(len(pmf)):
        if pmf[t] < threshold:
            new_w_func[t] = np.nan
    return new_w_func


def get_protocol_rate(parameters):
    """
    Returns the secret key rate for the input parameters.
    """
    print(f"\nRunning: {parameters}")
    pmf, w_func = repeater_sim(parameters)
    return secret_key_rate(pmf, w_func, parameters["t_trunc"])


def get_protocol(number_of_swaps, number_of_dists, where_to_distill=None):
    """
    Returns the protocol to be tested based on the input parameters.
    
    Parameters:
    - number_of_swaps (int): The number of swaps to be performed.
    - number_of_distillation (int): The number of distillation to be performed.
    - where_to_distill (int): The level at which to perform distillation.
    
    Returns:
    - tuple: the protocol to be tested.
    """
    distillations = [1] * number_of_dists
    swaps = [0] * number_of_swaps

    if number_of_dists == 0:
        protocol = swaps
    else:
        protocol = swaps[:where_to_distill] + distillations + swaps[where_to_distill:]
    
    return tuple(protocol)


def get_t_trunc(p_gen, p_swap, t_coh, swaps, dists, epsilon=0.01):
    """
    TODO: this is very unprecise and should be improved, especially in the distillation case.
    Returns the truncation time based on a lower bound of what is sufficient to reach 99% of the simulation cdf.
    """
    t_trunc = int((2/p_swap)**(swaps) * (1/p_gen) * (1/epsilon)) # not considering distillation

    p_dist = 0.5 # in the worst case, p_dist will never go below 0.5
    t_trunc *= (1/p_dist)**(dists) # considering distillation, but very unprecise

    t_trunc = min(max(t_coh, t_trunc//10), t_coh * 300)
    return int(t_trunc)


def sim_distillation_strategies(parameters):
    """
        Fixed parameters:
            - number of swaps
            - hardware paramters

        The function tests the performance of different distillation strategies,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["t_coh"],
                                        parameters["protocol"].count(0), parameters["protocol"].count(1))

    print(f"\nRunning: {parameters}")
    pmf, w_func = repeater_sim(parameters)

    rate = secret_key_rate(pmf, w_func, parameters["t_trunc"])
    print(f"Protocol {parameters['protocol']},\t r = {rate}\n")
    return rate, pmf, w_func


def single_test(): 
    """
    This function is used to test the repeater simulation with a fixed set of parameters.
     You can call this function from command line:
     ```py -c "from distillation_ml_gp import single_test; single_test()"```
    """
    parameters = {
                    "t_coh": 10000,
                    "p_gen": 0.999,
                    "p_swap": 0.999,
                    "w0": 0.999,
                    "t_trunc": 10000*300
                }
    parameters["protocol"] = (0,0,0,0,1,1,1,1,1,1,1,1,1,1,0)
    print(get_protocol_rate(parameters))    


class ThresholdExceededError(Exception):
    """
    This exception is raised when the CDF coverage is below the threshold.
    """
    def __init__(self, message="CDF under threshold count incremented", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info


def write_results(filename, parameters, best_results):
    """
    Write optimization results to file.
    """
    with open(filename, 'a') as file:
        output = (
                    f"\nProtocol parameters:\n"
                    f"{{\n"
                    f"    't_coh': {parameters['t_coh']},\n"
                    f"    'p_gen': {parameters['p_gen']},\n"
                    f"    'p_swap': {parameters['p_swap']},\n"
                    f"    'w0': {parameters['w0']}\n"
                    f"}}\n\n")
            
        for number_of_swaps, (best_protocol, best_score, error_coverage) in best_results.items():
            if best_protocol is None:
                logging.warning(
                        "Maximal number of attempts arrived. "
                        "Optimization fails.")
                print(f"CDF under threshold for {number_of_swaps} swaps")
                output += (
                        f"CDF under threshold for {number_of_swaps} swaps:\n"
                        f"    CDF coverage: {error_coverage*100}%\n\n"
                    )
            else:
                output += (
                        f"Best configuration for {number_of_swaps} swaps:\n"
                        f"    Best Protocol: {best_protocol}\n"
                        f"    Best secret key rate: {best_score}\n\n"
                    )
            
        file.write(output)


def plot_results(results, parameters, number_of_swaps, title, 
                 maximum: Tuple[np.float64, Tuple[int]], maxima: List[Tuple[np.float64, Tuple[int]]]):
    """
    This function plots the results of the brute force optimization.
    It creates a scatter plot where the y-axis represents the secret key rate,
    and the x-axis represents the protocol. X-axis ticks only show protocols starting with (0,.
    """
    key_rates = [result[0] for result in results]
    protocols = [result[1] for result in results]

    protocol_labels = [str(protocol) for protocol in protocols]
    
    best_protocol = maximum[1]
    best_protocol_label = str(best_protocol)

    plt.figure(figsize=(24, 8))
    plt.scatter(protocol_labels, key_rates, color='b', marker='o')
    plt.plot(protocol_labels, key_rates, color='b', linestyle='-', label='Secret Key Rate')
    
    to_label = True
    for _, maxima_protocol in maxima:
        if to_label:
            plt.axvline(x=str(maxima_protocol), color='g', linestyle='--', label='Local Maxima')
            to_label = False
        else:
            plt.axvline(x=str(maxima_protocol), color='g', linestyle='--')
    
    plt.axvline(x=best_protocol_label, color='r', linestyle='--', label='Global Maximum')

    plt.title(title, pad=20)
    plt.xlabel('Protocol')
    plt.ylabel('Secret Key Rate')
    plt.legend()

    maxima_labels = [str(maxima_protocol) for _, maxima_protocol in maxima]
    
    # Show only protocols ending with all swaps (zeros)
    if number_of_swaps != 1:
        plt.xticks(
            ticks = [i for i, p in enumerate(protocol_labels) 
                   if list(ast.literal_eval(p))[-number_of_swaps:] == [0]*number_of_swaps],
            labels = [p for i, p in enumerate(protocol_labels) 
                   if list(ast.literal_eval(p))[-number_of_swaps:] == [0]*number_of_swaps],
        )
    else:
        plt.xticks(ticks = range(len(protocol_labels)), labels = protocol_labels)
    
    for i, txt in enumerate(protocol_labels):
        if txt in maxima_labels:
            plt.text(i, key_rates[i]*1.035, txt, fontsize=9, ha='center', c='g')

    plt.gcf().autofmt_xdate()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.225)
    plt.savefig(f'{parameters["w0"]}_{number_of_swaps}_swaps_bests.png')


def get_all_maxima(ordered_results: List[Tuple[np.float64, Tuple[int]]], min_dists, max_dists) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    Get all the maxima from the results: the secret key rate and the protocol
        for the best secret key rate for each distillation number
    """
    maxima = []
    for number_of_dists in range(min_dists, max_dists + 1):
        maximum = max([result for result in ordered_results if result[1].count(1) == number_of_dists], key=lambda x: x[0])
        # Do not append if the maximum is zero
        if maximum[0] > 0:
            maxima.append(maximum)
    return maxima


def plot_process(min_dists, max_dists, parameters, number_of_swaps, 
                 results: List[Tuple[np.float64, Tuple[int]]], gp_result:OptimizeResult=None):
    """
    Invoke skopt plot functions to visualize the optimization results.
    """
    ordered_results: List[Tuple[np.float64, Tuple[int]]] = sorted(results, key=lambda x: x[0], reverse=True)

    maximum: Tuple[np.float64, Tuple[int]]  = (ordered_results[0][0], ordered_results[0][1])
    maxima: List[Tuple[np.float64, Tuple[int]]] = get_all_maxima(ordered_results, min_dists, max_dists)

    title = (
        f"Protocols with {number_of_swaps} swap{'' if number_of_swaps==1 else 's'}, "
        f"from {min_dists} to {max_dists} distillations\n"
        f"$p_{{gen}} = {parameters['p_gen']}, "
        f"p_{{swap}} = {parameters['p_swap']}, "
        f"w_0 = {parameters['w0']}, "
        f"t_{{coh}} = {parameters['t_coh']}$"
        f"\nBest secret-key-rate: {maximum[0]:.6f}, Protocol: {maximum[1]}"
    )   

    if gp_result is not None:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        plot_convergence(gp_result, ax=ax1)
        plot_objective(gp_result, ax=ax2)

        plt.tight_layout()
        plt.subplots_adjust(top=0.75, wspace=0.25)

        fig.suptitle(title)
        fig.savefig(f'{parameters["w0"]}_{number_of_swaps}_swaps_gp.png')
    
    plot_results(results, parameters, number_of_swaps, title, maximum, maxima)


def get_no_of_permutations_per_swap(min_dists, max_dists, s):
    """
    Returns the total number of permutations to be tested for a fixed number of swaps.
    """
    return int(sum([binom(s + d, d) for d in range(min_dists, max_dists + 1)]))


def get_analytical_permutations(min_dists, max_dists, min_swaps, max_swaps):
    """
    Returns the total number of permutations to be tested.
    """
    total_permutations = 0
    for s in range(min_swaps, max_swaps + 1):
        permutations = get_no_of_permutations_per_swap(s, min_dists, max_dists)
        total_permutations += permutations
    return total_permutations


def get_permutation_space(min_dists, max_dists, number_of_swaps, skopt_space=False):
    """
    The permutation space is used to test all the possible combinations of distillations for a fixed number of swaps.
    For each number of distillation tested, we test all the possible permutations of distillations. 
        i.e. for 2 swaps and from 0 to 2 distillations, we test:
        - zero distillations
            (0, 0), 
        - one distillation
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
        - two distillations
            (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)
    """
    space = OrderedDict()
    for number_of_dists in range(min_dists, max_dists + 1):
        distillations = [1] * number_of_dists
        swaps = [0] * number_of_swaps
        for perm in itertools.permutations(distillations + swaps):
            if perm.count(1) == number_of_dists:
                space[perm] = None
    space = list(space.keys())

    
    analytical_permutations = get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps)
    assert len(space) == analytical_permutations, f"Expected max. {analytical_permutations} permutations, got {len(space)}"
    if skopt_space:
        space = [
            Categorical([''.join(str(tup)) for tup in space], name='protocol')
        ]
    return space


def brute_force_optimization(parameters_set, space_type, min_swaps, max_swaps, min_dists, max_dists, filename):
    """
    This function is used to test the performance of different distillation strategies
    by bruteforcing all the possible configurations and returning the maximum rate achieved.
    """
    with open(filename, 'w') as file:
        file.write(f"From {min_swaps} to {max_swaps} swaps\nFrom {min_dists} to {max_dists} distillations\n")
        file.write(f"Bruteforce process for all the possible evaluations\n\n")

    for _, parameters in enumerate(parameters_set):
        best_results = {}

        for number_of_swaps in range(min_swaps, max_swaps+1):
            print(f"\n\nNumber of swaps: {number_of_swaps}")
            results: List[Tuple[np.float64, Tuple[int]]] = []

            space = []
            if space_type == "one_level":
                for number_of_dists in range(min_dists, max_dists+1):
                    for where_to_distill in range(number_of_swaps+1):
                        space.append(get_protocol(number_of_swaps, number_of_dists, where_to_distill))
            elif space_type == "enumerate":
                space = get_permutation_space(min_dists, max_dists, number_of_swaps)
            else: 
                raise ValueError("Invalid space")
            
            for protocol in space:
                try:
                    parameters["protocol"] = protocol
                    secret_key_rate, pmf, _ = sim_distillation_strategies(parameters)
                    
                    cdf_coverage = pmf_to_cdf(pmf)[-1]
                    if cdf_coverage < cdf_threshold:
                        raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})
                    
                    results.append((secret_key_rate, protocol))
                
                except ThresholdExceededError as e:
                    best_results[number_of_swaps] = (None, None, e.extra_info['cdf_coverage'])
                    break
                
            plot_process(min_dists, max_dists, parameters, number_of_swaps, results)

            ordered_results: List[Tuple[np.float64, Tuple[int]]] = sorted(results, key=lambda x: x[0], reverse=True)
            best_results[number_of_swaps] = (ordered_results[0][0], ordered_results[0][1], None)

        write_results(filename, parameters, best_results)

def objective_key_rate(space, space_type, number_of_swaps, parameters):
    """
    Objective function, consider the whole space of actions,
        returning a negative secret key rate
        in order for the optimizer to maximize the function.
    """
    # TODO: if the point has already been evaluated, evaluate another one (cache)
    # TODO: if all points have been evaluated, terminate the simulation

    if space_type == "one_level":
        number_of_dists = space['rounds of distillation']
        where_to_distill = space['after how many swaps we distill']
        parameters["protocol"] = get_protocol(
                                    number_of_swaps=number_of_swaps, 
                                    number_of_dists=number_of_dists, 
                                    where_to_distill=where_to_distill)
    
    elif space_type == "enumerate":
        parameters["protocol"] = ast.literal_eval(space['protocol'])
    
    secret_key_rate, pmf, _ = sim_distillation_strategies(parameters)
    
    cdf_coverage = pmf_to_cdf(pmf)[-1]
    if cdf_coverage < cdf_threshold:
        raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})                
    
    # The gaussian process minimizes the function, so return the negative of the key rate 
    return -secret_key_rate


def get_ordered_results(result: OptimizeResult, space_type) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    This function adjust the results to be positive and returns an ordered list of (key_rate, protocol)
    """
    result.fun = -result.fun
    result.func_vals = [-val for val in result.func_vals]
    
    assert len(result.x_iters) == len(result.func_vals)
    
    if space_type == "one_level":
        # TODO: use get_protocol here
        result_tuples = [(result.func_vals[i], result.x_iters[i]) for i in range(len(result.func_vals))]
    elif space_type == "enumerate":
        result_tuples = [(result.func_vals[i], ast.literal_eval(result.x_iters[i][0])) for i in range(len(result.func_vals))]

    ordered_results = sorted(result_tuples, key=lambda x: x[0], reverse=True)
    return ordered_results


def gaussian_optimization(parameters_set, space_type, min_swaps, max_swaps, min_dists, max_dists, gp_shots, gp_initial_points, filename):
    """
    This function is used to test the performance of different distillation strategies in an extensive way, 
    by using a Gaussian Process to optimize 
        - the number of distillations
        - the nesting level at which is applied.
    """
    with open(filename, 'w') as file:
        file.write(f"From {min_swaps} to {max_swaps} swaps\nFrom {min_dists} to {max_dists} distillations\n")
        file.write(f"Gaussian process with {gp_shots} evaluations and {gp_initial_points} initial points\n\n")

    for _, parameters in enumerate(parameters_set): 
        best_results = {}

        for number_of_swaps in range(min_swaps, max_swaps+1):
            print(f"\n\nNumber of swaps: {number_of_swaps}")
            
            if space_type == "one_level":
                space = [
                    Integer(min_dists, max_dists, name='rounds of distillation'), 
                    Integer(0, number_of_swaps, name='after how many swaps we distill'),
                ]
                @use_named_args(space)
                def wrapped_objective(**space_params):
                    return objective_key_rate(space_params, space_type, number_of_swaps, parameters)
            
            elif space_type == "enumerate":
                space = get_permutation_space(min_dists, max_dists, number_of_swaps, skopt_space=True)
                @use_named_args(space)
                def wrapped_objective(**space_params):
                    return objective_key_rate(space_params, space_type, number_of_swaps, parameters)
            
            else: 
                raise ValueError("Invalid space")

            try:
                if gp_shots is None:
                    gp_shots_def = gp_initial_points + get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps)//10

                result: OptimizeResult = gp_minimize(wrapped_objective, space, 
                                                     n_calls=(gp_shots or gp_shots_def), 
                                                     n_initial_points=gp_initial_points)
                
                ordered_results: List[Tuple[np.float64, Tuple[int]]] = get_ordered_results(result, space_type)

                plot_process(min_dists, max_dists, parameters, number_of_swaps, ordered_results, result)
            
            except ThresholdExceededError as e:
                best_results[number_of_swaps] = (None, None, e.extra_info['cdf_coverage'])
                continue
                
            # Get the best parameters and score from results
            best_results[number_of_swaps] = (ordered_results[0][0], ordered_results[0][1], None)
        
        write_results(filename, parameters, best_results)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=1, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=1, help="Maximum number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=7, help="Maximum amount of distillations to be performed")
    parser.add_argument("--optimizer", type=str, default="bf", help="Optimizer to be used")
    parser.add_argument("--space", type=str, default="enumerate", help="Space to be tested")
    parser.add_argument("--gp_shots", type=int, help="Number of shots for Gaussian Process")
    parser.add_argument("--gp_initial_points", type=int, default=50, help="Number of initial points for Gaussian Process")
    parser.add_argument("--filename", type=str, default='output.txt', help="Filename for output")
    
    args: Namespace = parser.parse_args()

    min_swaps: int = args.min_swaps
    max_swaps: int = args.max_swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists
    optimizer: str = args.optimizer
    gp_shots: int = args.gp_shots
    gp_initial_points: int = args.gp_initial_points
    filename: str = args.filename
    space: str = args.space

    cdf_threshold = 0.99

    parameters_set = [
        {
            't_coh': 10**4,
            'p_gen': 0.5,
            'p_swap': 0.5,
            'w0': 0.867
        },
        {
            't_coh': 10**4,
            'p_gen': 0.5,
            'p_swap': 0.5,
            'w0': 0.9
        },
        {
            't_coh': 10**4,
            'p_gen': 0.5,
            'p_swap': 0.5,
            'w0': 0.98
        },
        {
            't_coh': 10**4,
            'p_gen': 0.5,
            'p_swap': 0.5,
            'w0': 1.0
        },
    ]

    # TODO: encode "optimizer" parameter in "shots" parameter
    # which means, if the gp shots are enough to bruteforce the solutions, use a bf algorithm
    # allow also a value "-1", which means bf regardless of the shots
    if optimizer == "gp":
        gaussian_optimization(parameters_set, space, min_swaps, max_swaps, min_dists, max_dists, gp_shots, gp_initial_points, filename)
    elif optimizer == "bf":
        brute_force_optimization(parameters_set, space, min_swaps, max_swaps, min_dists, max_dists, filename)
    else:
        raise ValueError("Invalid optimizer")