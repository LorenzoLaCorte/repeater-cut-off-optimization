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
from collections import defaultdict

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
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
from skopt.space import Integer, Categorical, Real
from scipy.optimize import OptimizeResult
from skopt.utils import use_named_args

from distillation_gp_plots import plot_objective, plot_convergence

from repeater_algorithm import RepeaterChainSimulation
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


def get_protocol_space_size(space_type, min_dists, max_dists, number_of_swaps):
    """
    Returns the total number of protocols to be tested 
        for a specific space type and number of swaps.
    """
    if space_type == "one_level":
        return (max_dists - min_dists) * (number_of_swaps + 1) + (1 if min_dists == 0 else 0)
    elif space_type == "enumerate" or space_type == "strategy":
        return get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps)
    else:
        raise ValueError("Invalid space")
    

def get_protocol_rate(parameters):
    """
    Returns the secret key rate for the input parameters.
    """
    print(f"\nRunning: {parameters}")
    pmf, w_func = simulator.nested_protocol(parameters)
    return secret_key_rate(pmf, w_func, parameters["t_trunc"])


def sample_outcome(strategy, strategy_weight, protocol, idx, dists_target):
    """
    Returns 0 (swap) or 1 (distillation) based on the input parameters.

    Examples:
        Scenario 1: I miss 3 slots and I have 3 distillations missing
            Thus, I distill with p = 100% 
            (i.e., the first coin is depending on what is missing: 3 distillations / 3 slots)
    
        Scenario 2: I still have 3 slots and I have 1 distillation missing
            First of all, I throw a coin
            With 1/3 probability I distill
            While, the 2/3 remaining are distributed as it follows:
            - if I am dists_first, I distill with a prob. of strategy_weight
            - if I am swaps_first, I swap with a prob. of strategy_weight 
            - if I am alternate, 
                - if the previous was swap, I distill with a prob. of strategy_weight
                - if the previous was distill, I swap with a prob. of strategy_weight
    """
    dists_so_far = protocol.count(1)
    dists_remaining = dists_target - dists_so_far
    slots_remaining = len(protocol) - idx


    # Modeling the impact of the distillations applied so far on the decision
    if dists_remaining == 0:
        return 0
    
    dist_prob = dists_remaining/slots_remaining
    first_coin = np.random.choice([0, 1], p=[1 - dist_prob, dist_prob])
    if first_coin == 1:
        return 1
    else: # Modeling the impact of the strategies on the decision
        if strategy == "dists_first":
            return np.random.choice([0, 1], p=[1 - strategy_weight, strategy_weight])
        
        elif strategy == "swaps_first":
            return np.random.choice([0, 1], p=[strategy_weight, 1 - strategy_weight])
        
        elif strategy == "alternate":
            if idx == 0:
                return np.random.choice([0, 1])
            if protocol[idx-1] == 1:
                return np.random.choice([0, 1], p=[1 - strategy_weight, strategy_weight])
            if protocol[idx-1] == 0:
                return np.random.choice([0, 1], p=[strategy_weight, 1 - strategy_weight])

        else:
            raise ValueError("Invalid strategy")
        

def get_protocol_from_strategy(strategy, strategy_weight, number_of_swaps, number_of_dists):
    """
    Returns the protocol to be tested based on the input parameters.
    """
    protocol = [None] * (number_of_swaps + number_of_dists)

    for idx, _ in enumerate(protocol):
        protocol[idx] = int(sample_outcome(strategy, strategy_weight, protocol, idx, number_of_dists))

    assert protocol.count(1) == number_of_dists, f"Expected {number_of_dists} distillations, got {protocol.count(1)}"
    return tuple(protocol)


def get_protocol_from_distillations(number_of_swaps, number_of_dists, where_to_distill=None):
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
    This function is derived from Brand et. al, with an adjustment for distillation.
    TODO: it is a very lossy bound, it should be improved, to get the simulation going faster (mantaining cdf coverage).
    Returns the truncation time based on a lower bound of what is sufficient to reach (1-epsilon) of the simulation cdf.
    """
    t_trunc = int((2/p_swap)**(swaps) * (1/p_gen) * (1/epsilon)) # not considering distillation

    p_dist = 0.5 # in the worst case, p_dist will never go below 0.5
    t_trunc *= (1/p_dist)**(dists) # considering distillation, but very unprecise

    # Introduce a factor to reduce the truncation time, as the previous bound is very lossy
    reduce_factor = 10
    t_trunc = min(max(t_coh, t_trunc//reduce_factor), t_coh * 300)
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
    if fixed_t_trunc is not None:
        parameters["t_trunc"] = fixed_t_trunc
    else:
        parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["t_coh"],
                                            parameters["protocol"].count(0), parameters["protocol"].count(1))

    print(f"\nRunning: {parameters}")
    pmf, w_func = simulator.nested_protocol(parameters)

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
        't_coh': 35000,
        'p_gen': 0.002,
        'p_swap': 0.5,
        'w0': 0.97,
        "t_trunc": 350000
    }
    
    parameters["protocol"] = (1,1,1,0,0,0)
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
                 maximum: Tuple[np.float64, Tuple[int]], maxima: List[Tuple[np.float64, Tuple[int]]],
                 is_gp=False):
    """
    This function plots the results of the brute force optimization.
    It creates a scatter plot where the y-axis represents the secret key rate,
    and the x-axis represents the protocol. X-axis ticks only show protocols starting with (0,.
    """
    optimizer = "gp" if is_gp else "bf"

    tmp = results.copy()
    results = list(set(results))
    results = sorted(results, key=lambda x: (-len(x[1]), x[1]), reverse=True)

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
    plt.savefig(f'{parameters["w0"]}_{number_of_swaps}_swaps_{optimizer}.png')


def get_all_maxima(ordered_results: List[Tuple[np.float64, Tuple[int]]], min_dists, max_dists) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    Get all the maxima from the results: the secret key rate and the protocol
        for the best secret key rate for each distillation number
    """
    maxima = []
    for number_of_dists in range(min_dists, max_dists + 1):
        evaluations = [result for result in ordered_results if result[1].count(1) == number_of_dists]
        # Skip if there are no evaluations for the number of distillations
        if len(evaluations) == 0:
            continue
        maximum = max(evaluations, key=lambda x: x[0])
        # Do not append if the maximum is zero
        if maximum[0] > 0:
            maxima.append(maximum)
    return maxima


def plot_process(min_dists, max_dists, parameters, number_of_swaps, 
                 results: List[Tuple[np.float64, Tuple[int]]], gp_result:OptimizeResult=None):
    """
    Invoke skopt plot functions to visualize the optimization results.
    """
    is_gp = gp_result is not None
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

    if is_gp:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        plot_convergence(gp_result, ax=ax1)
        plot_objective(gp_result, ax=ax2)

        plt.tight_layout()
        plt.subplots_adjust(top=0.75, wspace=0.25)

        fig.suptitle(title)
        fig.savefig(f'{parameters["w0"]}_{number_of_swaps}_swaps_skopt.png')
    
    plot_results(results, parameters, number_of_swaps, title, maximum, maxima, is_gp)


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
                if min_dists == 0:
                    space.append(tuple([0]*number_of_swaps))
                for number_of_dists in range(max(1, min_dists), max_dists+1):
                    for where_to_distill in range(number_of_swaps+1):
                        space.append(get_protocol_from_distillations(number_of_swaps, number_of_dists, where_to_distill))
            elif space_type == "enumerate":
                space = get_permutation_space(min_dists, max_dists, number_of_swaps)
            elif space_type == "strategy":
                raise ValueError("Bruteforce not supported for strategy space, use 'enumerate'")
            else: 
                raise ValueError("Invalid space")
            
            protocol_space_size = get_protocol_space_size(space_type, min_dists, max_dists, number_of_swaps)
            assert len(space) == protocol_space_size, \
                f"Expected {protocol_space_size} protocols, got {len(space)}"

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


# Cache results of the objective function to avoid re-evaluating the same point and speed up the optimization
cache_results = defaultdict(np.float64)
strategy_to_protocol = {}

def objective_key_rate(space, space_type, number_of_swaps, parameters):
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
    
    elif space_type == "enumerate":
        parameters["protocol"] = ast.literal_eval(space['protocol'])
    
    elif space_type == "strategy":
        number_of_dists = space['rounds of distillation']
        strategy = space['strategy']
        strategy_weight = space['strategy weight']
        parameters["protocol"] = get_protocol_from_strategy(strategy, strategy_weight, number_of_swaps, number_of_dists)
        strategy_to_protocol[(number_of_dists, strategy, strategy_weight)] = parameters["protocol"]

    else:
        raise ValueError("Invalid space")
    
    if parameters["protocol"] in cache_results:
        logging.warning("Already evaluated protocol, returning cached result")
        return -cache_results[parameters["protocol"]]
    
    secret_key_rate, pmf, _ = sim_distillation_strategies(parameters)
    
    cdf_coverage = pmf_to_cdf(pmf)[-1]
    if cdf_coverage < cdf_threshold:
        raise ThresholdExceededError(extra_info={'cdf_coverage': cdf_coverage})                

    cache_results[parameters["protocol"]] = secret_key_rate
    space.update({'protocol': parameters["protocol"]})
    # The gaussian process minimizes the function, so return the negative of the key rate 
    return -secret_key_rate


def get_ordered_results(result: OptimizeResult, space_type, number_of_swaps) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    This function adjust the results to be positive and returns an ordered list of (key_rate, protocol)
    """
    result.fun = -result.fun
    result.func_vals = [-val for val in result.func_vals]
    assert len(result.x_iters) == len(result.func_vals)
    
    if space_type == "one_level":
        result_tuples = [(  result.func_vals[i], 
                            get_protocol_from_distillations(
                                number_of_swaps=number_of_swaps,
                                number_of_dists=result.x_iters[i][0], 
                                where_to_distill=result.x_iters[i][1])
                         )  
                        for i in range(len(result.func_vals))]

    elif space_type == "enumerate":
        result_tuples = [(result.func_vals[i], ast.literal_eval(result.x_iters[i][0])) for i in range(len(result.func_vals))]

    # BUG: if I call this function, it is not deterministic, I have to store the information about the protocol in result struct
    elif space_type == "strategy":
        result_tuples = [(result.func_vals[i], result.x_iters[i][-1]) for i in range(len(result.func_vals))]

    ordered_results = sorted(result_tuples, key=lambda x: x[0], reverse=True)
    return ordered_results


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
        logging.warning(f"All protocols evaluated ({len(cache_results)}/{protocol_space_size}), stopping optimization")
        
        if len(result.models) == 0:
            logging.warning(f"Evaluation terminated within initial points: protocol space is too small or initial points are too many")
            result.models = [GaussianProcessRegressor] # Use a random model for the partial dependence plot
        return True
    

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
            
            global protocol_space_size # TODO: refactor in order to avoid using global variables
            protocol_space_size = get_protocol_space_size(space_type, min_dists, max_dists, number_of_swaps)

            if space_type == "one_level":
                space = [
                    Integer(min_dists, max_dists, name='rounds of distillation'), 
                    Integer(0, number_of_swaps, name='after how many swaps we distill'),
                ]
                @use_named_args(space)
                def wrapped_objective(**space_params):
                    return objective_key_rate(space_params, space_type, number_of_swaps, parameters)
            
            elif space_type == "strategy":
                space = [
                    Integer(min_dists, max_dists, name='rounds of distillation'), 
                    Categorical(["dists_first", "swaps_first", "alternate"], name='strategy'),
                    Real(0.5, 1.0, name='strategy weight'),
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

            # Define reasonable default values (in terms of a percentage of the protocol space size)
            if gp_initial_points is None:
                gp_initial_points_def = 20 + int(protocol_space_size * .10)
            if gp_shots is None:
                gp_shots_def = (gp_initial_points or gp_initial_points_def) + int(protocol_space_size * .20)

            try:
                # TODO: potentially, I think it maybe a good idea to give all-dists-first as initial points, but it maybe introduce bias
                # Give only swap protocol as initial point
                x0 = [
                    0, 
                    "dists_first",
                    1.0,
                ]

                # Perform the optimization
                result: OptimizeResult = gp_minimize(wrapped_objective, space, 
                                                     n_calls=(gp_shots or gp_shots_def), 
                                                     n_initial_points=(gp_initial_points or gp_initial_points_def),
                                                     callback=[is_gp_done],
                                                     x0=x0,
                                                     acq_func='LCB',
                                                     kappa=1.96*2, # Double the default kappa, to prefer exploration
                                                     noise=1e-10, # There is no noise in results
                                                     ) 
                
                ordered_results: List[Tuple[np.float64, Tuple[int]]] = get_ordered_results(result, space_type, number_of_swaps)
                plot_process(min_dists, max_dists, parameters, number_of_swaps, ordered_results, result)
                cache_results.clear()

            except ThresholdExceededError as e:
                best_results[number_of_swaps] = (None, None, e.extra_info['cdf_coverage'])
                continue          
        
            # Get the best parameters and score from results
            best_results[number_of_swaps] = (ordered_results[0][0], ordered_results[0][1], None)
        
        write_results(filename, parameters, best_results)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=1, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=3, help="Maximum number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=7, help="Maximum amount of distillations to be performed")
    
    parser.add_argument("--optimizer", type=str, default="bf", help="Optimizer to be used")
    parser.add_argument("--space", type=str, default="enumerate", help="Space to be tested")
    parser.add_argument("--gp_shots", type=int, help="Number of shots for Gaussian Process")
    parser.add_argument("--gp_initial_points", type=int, help="Number of initial points for Gaussian Process")
    parser.add_argument("--filename", type=str, default='output.txt', help="Filename for output")
    
    parser.add_argument("--t_coh", type=int, nargs='+', default=[120], help="Coherence times")
    parser.add_argument("--p_gen", type=float, nargs='+', default=[0.9], help="Generation probabilities")
    parser.add_argument("--p_swap", type=float, nargs='+', default=[0.9], help="Swap probabilities")
    parser.add_argument("--w0", type=float, nargs='+', default=[0.933], help="Initial weights")

    parser.add_argument("--dp", action='store_true', help="Use dynamic programming to cache results and set a fixed truncation time")

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

    t_coh = args.t_coh
    p_gen = args.p_gen
    p_swap = args.p_swap
    w0 = args.w0

    dp_enabled = args.dp
    global fixed_t_trunc
    global simulator
    simulator = None

    if dp_enabled:
        simulator = RepeaterChainSimulation(use_cache=True)
        fixed_t_trunc = get_t_trunc(
            min(p_gen), 
            min(p_swap),
            min(t_coh),
            max_swaps, max_dists)
    else:
        fixed_t_trunc = None
        simulator = RepeaterChainSimulation(use_cache=False)
    
    cdf_threshold = 0.99

    parameters_set = [
        {
            't_coh': t_coh[i],
            'p_gen': p_gen[i],
            'p_swap': p_swap[i],
            'w0': w0[i]
        } for i in range(len(t_coh))
    ]

    # If the gp shots are enough to bruteforce the solutions, use a bf algorithm
    if gp_shots is not None and gp_shots >= get_protocol_space_size(space, min_dists, max_dists, max_swaps):
        logging.warning("GP shots are enough to bruteforce the solutions, using brute force algorithm")
        optimizer = "bf"
        gp_shots = None
    
    # Start the optimization process
    if optimizer == "gp":
        gaussian_optimization(parameters_set, space, min_swaps, max_swaps, min_dists, max_dists, gp_shots, gp_initial_points, filename)
    elif optimizer == "bf":
        brute_force_optimization(parameters_set, space, min_swaps, max_swaps, min_dists, max_dists, filename)
    else:
        raise ValueError("Invalid optimizer")