"""
This script is used to run simulations to compare different quantum repeater protocols 
and visualize the performance of these protocols under various conditions. 
We perform some manual steps, calling the unit protocols, to understand the evolution of the distributions given in output
We also get some insights on the comparison between different paramters' regimes
"""

import matplotlib.pyplot as plt
colorblind_palette = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#000000",
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)

import copy
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)
from utility_functions import secret_key_rate, pmf_to_cdf, get_mean_werner, get_mean_waiting_time
from gp_utils import get_protocol_enum_space, load_config

from matplotlib.ticker import MaxNLocator
import itertools

from enum import Enum

config = load_config('config.json')

class DistillationType(Enum):
    """
    Enum class for the distillation type.
    """
    DISTILLATION_FIRST = (1, 0, 0, 0)
    SWAPPING_FIRST = (0, 1, 0, 0)
    NO_DISTILLATION = (0, 0, 0)


def get_protocol_rate(parameters):
    """
    Returns the secret key rate for the input parameters.
    """
    print(f"\nRunning: {parameters}")
    pmf, w_func = repeater_sim(parameters)
    return secret_key_rate(pmf, w_func, parameters["t_trunc"])


def single_asym_test(): 
    """
    This function is used to test the repeater simulation with a fixed set of parameters.
     You can call this function from command line:
     ```py -c "from distillation_tests.py import single_test; single_test()"```
    """
    parameters = {
        't_coh': 3600000,
        'p_gen': 0.0026,
        'p_swap': 0.85,
        'w0': 0.958,
        "t_trunc": 321168,
    }
    
    parameters["protocol"] = ('d0', 'd1', 's0', 'd2', 'd3', 's2', 'd1', 'd3', 's1', 'd4', 'd5', 's4', 'd3', 'd5', 's3', 'd6', 'd7', 's6', 'd8', 'd9', 's8', 'd7', 'd9', 's7', 'd5', 'd9', 's5')
    print(get_protocol_rate(parameters))


def single_test(): 
    """
    This function is used to test the repeater simulation with a fixed set of parameters.
     You can call this function from command line:
     ```py -c "from distillation_ml_gp import single_test; single_test()"```
    """
    # "1400000 0.00092     0.85 0.952 5  1"

    parameters = {
        't_coh': 1400000,
        'p_gen': 0.00092,
        'p_swap': 0.85,
        'w0': 0.952,
        "t_trunc": 1400000
    }
    
    parameters["protocol"] = (1,1,0,0)
    print(get_protocol_rate(parameters))


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
    """
    new_w_func = w_func.copy()
    for t in range(len(pmf)):
        if pmf[t] < threshold:
            new_w_func[t] = np.nan
    return new_w_func


def save_plot(fig, axs, row_titles, parameters={}, rate=None, exp_name="protocol.png", legend=False):
    """
    Formats the input figure and axes.
    """
    if axs.ndim == 2:
        rows, cols = axs.shape
        for i in range(rows):
            for j in range(cols):
                axs[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
            if legend:
                axs[i, -1].legend()
    else:  # axs is 1D
        for ax in axs:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if legend:
                ax.legend()

    title_params = f"p_gen = {parameters['p_gen']}, p_swap = {parameters['p_swap']}, w0 = {parameters['w0']}, t_coh = {parameters['t_coh']}"
    title = f"protocol with {exp_name.replace('_', ' ')} - {title_params}"
    if rate is not None:
        title += f" - R = {rate}"
    fig.suptitle(title)


    if row_titles is not None:
        for ax, row_title in zip(axs[:,0], row_titles):
            ax.text(-0.2, 0.5, row_title, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=14)

    left_space = 0.1 if row_titles is not None else 0.06
    plt.tight_layout()
    plt.subplots_adjust(left=left_space, top=(0.85 + axs.shape[0]*0.02), hspace=0.2)

    fig.savefig(f"{exp_name}.png")
    

def plot_pmf_cdf_werner(pmf, w_func, trunc, axs, row, full_werner=True, label=None):
    """
    Plots (on the input axs) the PMF, CDF, and Werner parameter arrays (one row of subplots),
    making deep copies of the input arrays to ensure the originals are not modified.
    """
    assert len(pmf) >= trunc, "The length of pmf must be larger or equal to t_trunc"

    pmf_copy = copy.deepcopy(pmf)[:trunc]
    w_func_copy = copy.deepcopy(w_func)[:trunc]

    w_func_copy[0] = np.nan
    cdf = pmf_to_cdf(pmf_copy)
    
    plot_data = {
        "PMF": pmf_copy,
        "CDF": cdf,
        "Werner parameter": remove_unstable_werner(pmf_copy, w_func_copy)
    }
    
    plot_axs = axs[row] if axs.ndim == 2 else axs  # handle both 1D and 2D axes arrays
    
    for title, data in plot_data.items():
        ax = plot_axs[list(plot_data.keys()).index(title)]
        if label is not None and title == "Werner parameter":
            ax.plot(np.arange(trunc), data, label=label)
        else:
            ax.plot(np.arange(trunc), data)
        ax.set_xlabel("Waiting Time")
        ax.set_title(title)
        if title == "Werner parameter":
            if full_werner:
                ax.set_ylim([0, 1])
            ax.set_ylabel("Werner parameter")            
        else:
            ax.set_ylabel("Probability")


def entanglement_distillation_runner(distillation_type, parameters):
    """
    Runs the entanglement distillation simulation with the specified parameters.
    """
    protocol = distillation_type.value
    parameters["protocol"] = protocol

    pmf, w_func = repeater_sim(parameters)
    return pmf, w_func


def distillation_step(parameters, pmf, w_func, pmf2= None, w_func2=None):
    """
    Distills the entanglement between two qubits pairs, given the input PMF and Werner parameter arrays.
    Returns the new PMF and Werner parameter arrays after distillation.
    """
    simulator = RepeaterChainSimulation()
    if pmf2 is not None and w_func2 is not None:
        new_pmf, new_w_func = simulator.compute_unit(
            parameters, pmf, w_func, pmf2, w_func2, unit_kind="dist")
    else:
        new_pmf, new_w_func = simulator.compute_unit(
            parameters, pmf, w_func, unit_kind="dist")
    return new_pmf, new_w_func


def swapping_step(parameters, pmf, w_func, pmf2=None, w_func2=None):
    """
    Swaps the entanglement between two qubits pairs, given the input PMF and Werner parameter arrays.
    Returns the new PMF and Werner parameter arrays after swapping.
    """
    simulator = RepeaterChainSimulation()
    if pmf2 is not None and w_func2 is not None:
        new_pmf, new_w_func = simulator.compute_unit(
            parameters, pmf, w_func, pmf2, w_func2, unit_kind="swap")
    else:
        new_pmf, new_w_func = simulator.compute_unit(
            parameters, pmf, w_func, unit_kind="swap")
    return new_pmf, new_w_func


def plot_steps(fig, axs, pmfs, w_funcs, trunc=None, label=None):
    assert len(pmfs) == len(w_funcs), "The number of PMF and Werner parameter arrays must be the same."
    rows = len(pmfs)

    if trunc is None:
        trunc = len(pmfs[0])

    for i in range(rows):
        if label is not None:
            plot_pmf_cdf_werner(pmfs[i], w_funcs[i], trunc, axs, i, full_werner=False, label=f"{label.replace('_', ' ')}, r = {secret_key_rate(pmfs[i], w_funcs[i]):.5f}")
        else:    
            plot_pmf_cdf_werner(pmfs[i], w_funcs[i], trunc, axs, i, full_werner=True)

def entanglement_distillation_manual(distillation_type, parameters, intermediate=False):
    """
    Runs the entanglement distillation experiment but by doing the steps one by one and plotting all the intermediate results.
    This example includes protocols with 
        - distillation first, (1, 0, 0, 0)
        - swapping first, (0, 1, 0, 0)
        - and no distillation (0, 0, 0).
    If intermediate is set, it will return the PMF and Werner parameter arrays after each step.
    """
    p_gen = parameters["p_gen"]
    w0 = parameters["w0"]
    t_trunc = parameters["t_trunc"]
    
    # Generate entanglement link for all qubits pairs between AB, BC and CD
    # All the links pmf and werner parameter are represented by the same arrays
    pmf_gen = np.concatenate(
        (np.array([0.]),  # generation needs at least one step.
        p_gen * (1 - p_gen)**(np.arange(1, t_trunc) - 1))
        )
    w_gen = w0 * np.ones(t_trunc)  # initial werner parameter

    if distillation_type == DistillationType.DISTILLATION_FIRST:
        # Distill the entanglement between A and B
        pmf_dist, w_dist = distillation_step(
            parameters, pmf_gen, w_gen)
        # And then perform the swapping between A and B
        pmf_swap1, w_swap1 = swapping_step(
            parameters, pmf_dist, w_dist)
    elif distillation_type == DistillationType.SWAPPING_FIRST:
        # Swap the entanglement between A and B
        pmf_swap_nodist, w_swap_nodist = swapping_step(
            parameters, pmf_gen, w_gen)
        # And then perform the distillation between A and B
        pmf_swap1, w_swap1 = distillation_step(
            parameters, pmf_swap_nodist, w_swap_nodist)
    elif distillation_type == DistillationType.NO_DISTILLATION:
        # Perform the swapping between A and B    
        pmf_swap1, w_swap1 = swapping_step(
            parameters, pmf_gen, w_gen)
    else:
        raise ValueError("Invalid distillation type")
    
    # We then perform two more swapping steps
    pmf_swap2, w_swap2 = swapping_step(
        pmf_swap1, w_swap1, parameters)
    pmf_swap3, w_swap3 = swapping_step(
        pmf_swap2, w_swap2, parameters)

    if intermediate: 
        if distillation_type == DistillationType.DISTILLATION_FIRST:
            return [pmf_gen, pmf_dist, pmf_swap1, pmf_swap2, pmf_swap3], [w_gen, w_dist, w_swap1, w_swap2, w_swap3]
        elif distillation_type == DistillationType.SWAPPING_FIRST:
            return [pmf_gen, pmf_swap_nodist, pmf_swap1, pmf_swap2, pmf_swap3], [w_gen, w_swap_nodist, w_swap1, w_swap2, w_swap3]
        elif distillation_type == DistillationType.NO_DISTILLATION:
            return [pmf_gen, pmf_swap1, pmf_swap1, pmf_swap2, pmf_swap3], [w_gen, w_swap1, w_swap1, w_swap2, w_swap3]
    else:
        return pmf_swap3, w_swap3


def compare_manual_algorithm():
    """
    Runs the entanglement distillation experiment with the specified parameters.
    """

    experiment = {"p_gen": 0.1, "p_swap": 0.4, "t_trunc": 5000, 
                  "t_coh": 800, "w0": 0.85, "dist_type": DistillationType.NO_DISTILLATION}

    pmf, w_func = entanglement_distillation_runner(experiment["dist_type"], experiment)
    pmf_man, w_func_man = entanglement_distillation_manual(experiment["dist_type"], experiment)

    rows = 2
    fig, axs = plt.subplots(rows, 3, figsize=(15, 4*rows), sharex=True)
    
    plot_pmf_cdf_werner(pmf, w_func, experiment["t_trunc"], axs, 0)
    plot_pmf_cdf_werner(pmf_man, w_func_man, experiment["t_trunc"], axs, 1)

    exp_name = experiment["dist_type"].name.lower() + "_manual_vs_algorithm"
    save_plot(fig=fig, axs=axs, row_titles=None, parameters=experiment, rate=secret_key_rate(pmf, w_func), exp_name=exp_name)


def check_distance(pmf1, pmf2, w_func1, w_func2, threshold=0.01):
    pmf_distance = np.linalg.norm(np.array(pmf1) - np.array(pmf2))
    w_func_distance = np.linalg.norm(np.array(w_func1) - np.array(w_func2))

    if pmf_distance > threshold or w_func_distance > threshold:
        print("Check failed: The arrays are more distant than the allowed threshold.")


def run_manual_experiment(parameters, dist_type=DistillationType.DISTILLATION_FIRST):
    """
    Runs the entanglement distillation experiment with the specified experiment parameters.
    """
    parameters["dist_type"] = dist_type

    pmf, w_func = entanglement_distillation_runner(parameters["dist_type"], parameters)
    pmf_man, w_func_man = entanglement_distillation_manual(parameters["dist_type"], parameters)

    check_distance(pmf, pmf_man, w_func, w_func_man)
    return pmf_man, w_func_man


def run_distillation_types():
    """
    Runs the entanglement distillation experiment with different distillation types.
    """
    parameters = {"p_gen": 0.1, "p_swap": 0.4, "t_trunc": 10000, 
                  "t_coh": 800, "w0": 0.85}

    rows = 1
    fig, axs = plt.subplots(rows, 3, figsize=(20, 5*rows), sharex=True)
    
    for dist_type in DistillationType:
        pmf, w_func = run_manual_experiment(parameters, dist_type)
        plot_pmf_cdf_werner(pmf, w_func, (parameters["t_trunc"]), axs, 0, full_werner=False, label=dist_type.name.lower())
    
    save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, rate=None,
              exp_name="distillation_types", legend=True)


def tuple_to_bitstring(tup):
    # Call recursively if it is a tuple of tuples
    if isinstance(tup[0], tuple):
        return ", ".join(tuple_to_bitstring(t) for t in tup)
    else:
        return "".join([str(i) for i in tup])


def plot_protocols(results, parameters, metric):
    """
    This function plots the results of the optimization.
    It creates a scatter plot where the y-axis represents the secret key rate,
    and the x-axis represents the protocol. X-axis ticks only show protocols starting with (0,.
    """
    # Transform the tuples of protocols into bit strings
    results = [(data, tuple_to_bitstring(protocol)) for data, protocol in results]
    results = sorted(results, key=lambda x: (len(results[1])), reverse=False)

    metric_values = [result[0] for result in results]
    protocols = [result[1] for result in results]

    protocol_labels = [str(protocol) for protocol in protocols]
    
    plt.figure(figsize=(config['figsize_key_rates']['width'], config['figsize_key_rates']['height']))
    plt.scatter(protocol_labels, metric_values, color='b', marker='o')
    plt.plot(protocol_labels, metric_values, color='b', linestyle='-', label=f'{metric}')
    

    plt.title(f"{metric} for Protocols in the Space", pad=20)
    plt.xlabel('Protocol')
    plt.ylabel(f'{metric}')
    plt.legend()

    # # Show only protocols ending with all swaps (zeros)
    # plt.xticks(
    #     ticks = [i for i, p in enumerate(protocol_labels) 
    #             if list(ast.literal_eval(p))[-number_of_swaps:] == [0]*number_of_swaps],
    #     labels = [p for i, p in enumerate(protocol_labels) 
    #             if list(ast.literal_eval(p))[-number_of_swaps:] == [0]*number_of_swaps],
    # )
    
    plt.gcf().autofmt_xdate()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.225)
    plt.savefig(f"dist_{str(metric).replace(' ', '').lower()}_protocols.png", dpi=config['high_dpi'])


def calculate_benchmark(parameters):
    def calculate_subblock_benchmark(parameters, protocol):
        parameters["protocol"] = protocol
        benchmark = {
            "pmf": repeater_sim(parameters)[0],
            "w_func": repeater_sim(parameters)[1]
        }
        return benchmark

    b1_benchmark = calculate_subblock_benchmark(parameters, (0,0))
    b2_benchmark = calculate_subblock_benchmark(parameters, (0,))

    bm_pmf, bm_w_func = swapping_step(parameters, b1_benchmark["pmf"], b1_benchmark["w_func"],
                              b2_benchmark["pmf"], b2_benchmark["w_func"])

    return {
        "pmf": bm_pmf,
        "w_func": bm_w_func
    }


def optimality_test_seven_nodes_protocols(parameters, min_dists, max_dists):
    """
    Runs the entanglement distillation experiment 
        with the specified parameters for the seven nodes protocol.

    We consider two blocks: one of five nodes (B1), intersecting with one of three nodes (B2)
        B1 performs two SWAPs to reach an entangled link
        B2 performs one SWAP  to reach an entangled link
        Then a further SWAP links the end nodes of the two blocks

    Distillation can be performed at each level for B1, B2 and before or after the SWAP linking the two
    
    Thus,
        for B1 we run: (0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,1,1), (0,1,0,1), (0,1,1,0), (1,0,0,1), (1,0,1,0), (1,1,0,0), ...
        i.e. the whole space of protocols for n=2 and k in the range from min_dists to max_dists]
        for B2 we run: (0), (0,1), (1,0), (0,1,1), (1,0,1), (1,1,0), ...
        the same, but for n=1
    
    We then test out the performance of all the possible combinations of these protocols
    Also, we can distill the entanglement between the two blocks, before or after the SWAP, or not at all (3 possibilities)
    """

    # Get the permutation spaces for the two blocks
    b1_space = [(0,0), (0,1,0), (1,0,0)] # get_protocol_enum_space(min_dists, max_dists, number_of_swaps=2)
    b2_space = [(0,), (1,0), (0,1)] # get_protocol_enum_space(min_dists, max_dists, number_of_swaps=1)

    b1_results = {}
    b2_results = {}

    def collect_block_results(protocol_space, parameters, results):
        for protocol in protocol_space:
            parameters["protocol"] = protocol
            pmf, w_func = repeater_sim(parameters)
            results[protocol] = {"pmf": pmf, "w_func": w_func}

    collect_block_results(b1_space, parameters, b1_results)
    collect_block_results(b2_space, parameters, b2_results)

    # Now, combine the results of the two blocks
    combined_results = {}
    for b1_protocol, b1_data in b1_results.items():
        for b2_protocol, b2_data in b2_results.items():
            pmf, w_func = swapping_step(parameters, b1_data["pmf"], b1_data["w_func"], b2_data["pmf"], b2_data["w_func"])
            combined_results[(b1_protocol, b2_protocol)] = {"pmf": pmf, "w_func": w_func}

    waiting_times = [(get_mean_waiting_time(data["pmf"]), protocol) for protocol, data in combined_results.items()]
    werner_params = [(get_mean_werner(data["pmf"], data["w_func"]), protocol) for protocol, data in combined_results.items()]

    plot_protocols(waiting_times, parameters, "Waiting Time")
    plot_protocols(werner_params, parameters, "Werner Parameter")

    benchmark = calculate_benchmark(parameters)

    # Compute the ratio of the average waiting time without distillation to the average waiting time with distillation
    waiting_times_improv = [(data/get_mean_waiting_time(benchmark["pmf"]), protocol) for data, protocol in waiting_times]
    werner_params_improv = [(data/get_mean_werner(benchmark["pmf"], benchmark["w_func"]), protocol) for data, protocol in werner_params]

    plot_protocols(waiting_times_improv, parameters, "Waiting Time Ratio (to Benchmark)")
    plot_protocols(werner_params_improv, parameters, "Werner Parameter Ratio (to Benchmark)")        


def hw_varying_experiment(hw_param_name: str, hw_param_val_range,
                          parameters, min_dists, max_dists, number_of_swaps=2):
    """
    Perform variant simulations with the hw parameter in range (param/gen_variants, param+param/gen_variants, param/gen_variants)
    and plots the results for the different protocols for different metrics
    """
    parameters = copy.deepcopy(parameters)
    protocol_space = get_protocol_enum_space(min_dists, max_dists, number_of_swaps)
    val_range = hw_param_val_range

    results = {}
    for protocol in protocol_space:
        results[protocol] = {}
        for hw_param_data in val_range:
            parameters["protocol"] = protocol
            parameters[f"{hw_param_name}"] = hw_param_data
            pmf, w_func = repeater_sim(parameters)

            results[protocol][f"{hw_param_data}"] = {
                "avg_waiting_time": get_mean_waiting_time(pmf),
                "avg_werner": get_mean_werner(pmf, w_func),
                "secret_key_rate": secret_key_rate(pmf, w_func)
            }

    metrics = {
        "avg_waiting_time": "Average Waiting Time",
        "avg_werner": "Average Werner Parameter",
        "secret_key_rate": "Secret Key Rate"
    }

    consider_improvement = True
    if consider_improvement:

        for protocol in protocol_space:
            no_dist_protocol = (0,) * protocol.count(0)
            for hw_param_data in val_range:
                for metric_abbr, metric_full in metrics.items():
                    no_dist_data = results[no_dist_protocol][f"{hw_param_data}"][metric_abbr]
                    with_dist_data = results[protocol][f"{hw_param_data}"][metric_abbr]
                    if metric_abbr == "avg_werner":
                        improv_data = (with_dist_data / no_dist_data) if no_dist_data != 0. else 0.
                    else:
                        improv_data = (no_dist_data / with_dist_data) if with_dist_data != 0. else 0.
                    results[protocol][f"{hw_param_data}"][f"{metric_abbr}_improv"] = improv_data

        metrics.update({f"{metric_abbr}_improv": f"{metric_full} Ratio (no-dist to dist)" for metric_abbr, metric_full in metrics.items()})
    
    num_dists = max_dists - min_dists + 1
    fig, axes = plt.subplots(1, num_dists, figsize=(config["figsize_hw_varying"]["width"], config["figsize_hw_varying"]["height"]), sharey=True)

    for metric_abbr, metric_full in metrics.items():
        
        for i, num_dist in enumerate(range(min_dists, max_dists + 1)):

            sub_results = {protocol: data for protocol, data in results.items() if protocol.count(1) == num_dist}

            ax = axes[i]
            for protocol, hw_param_data in sub_results.items():
                hw_param_points = list(hw_param_data.keys())
                metric_values = [data[metric_abbr] for data in hw_param_data.values()]
                ax.plot(hw_param_points, metric_values, marker='o', label=f"Protocol {protocol}")

            ax.set_title(f"{num_dist} distillation{'s' if num_dist != 1 else ''}")
            ax.set_xlabel(f"${hw_param_name.replace('0', '_0').replace('_', '_{')+'}'}$")
            if i == 0:
                ax.set_ylabel(metric_full)
            ax.legend()

        title = (
            f"{metric_full}\n"
            f"Protocols with $N = {1+2**number_of_swaps}$, i.e. {number_of_swaps} swap{'' if number_of_swaps==1 else 's'}, "
            f"and from {min_dists} to {max_dists} distillations\n"
            + (f"$p_{{gen}} = {parameters['p_gen']}$, " if hw_param_name != 'p_gen' else "varying $p_{gen}$, ")
            + (f"$p_{{swap}} = {parameters['p_swap']}$, " if hw_param_name != 'p_swap' else "varying $p_{swap}$, ")
            + (f"$w_0 = {parameters['w0']}$, " if hw_param_name != 'w0' else "varying $w_{0}$, ")
            + (f"$t_{{coh}} = {parameters['t_coh']}$\n" if hw_param_name != 't_coh' else "varying $t_{coh}$\n")
        )

        plt.suptitle(title)
        plt.legend()
        plt.tight_layout(w_pad=2, h_pad=0.5)
        plt.savefig(f"hw_{hw_param_name}_{metric_abbr}_protocols.png")
        
        for ax in axes:
            ax.clear()

def hw_varying_experiment_runner():
    # Set the parameters for the experiment
    t_coh = 8
    parameters = {"p_gen": 0.5, "p_swap": 0.5, "t_trunc": t_coh*300, 
                "t_coh": t_coh, "w0": 0.88}
    

    # Distillation steps experiment
    DISTILLATION_STEPS = False
    # Optimality test for seven nodes protocols
    OPTIMALITY_TEST = False
    # Hardware varying experiment
    HW_VARYING = True
    

    if DISTILLATION_STEPS:
        zoom = 10
        rows = 5
        fig, axs = plt.subplots(rows, 3, figsize=(15, 4*rows), sharex=True)

        for dist_type in DistillationType:
            pmfs, w_funcs = entanglement_distillation_manual(dist_type, parameters, intermediate=True)
            plot_steps(fig, axs, pmfs, w_funcs, (parameters["t_trunc"]//zoom), dist_type.name.lower())

        save_plot(fig, axs, None, parameters, rate=None, exp_name="distillation_steps", legend=True)

    if OPTIMALITY_TEST:
        optimality_test_seven_nodes_protocols(parameters, 0, 2, gen_variants=2)

    # Run the experiment with specific values range and harware parameters
    # or run the experiment with all hardware parameters
    SPECIFIC_EXPERIMENT = True

    if HW_VARYING:
        if SPECIFIC_EXPERIMENT:
            hw_param = "p_gen"
            hw_param_val_range = [np.round((0.1) * i, 1) for i in range(1, 11)]
            min_dists, max_dists, number_of_swaps = 0, 2, 2
            hw_varying_experiment(hw_param, hw_param_val_range, parameters, min_dists, max_dists, number_of_swaps)

        else:
            for hw_param in ["p_gen", "p_swap", "w0", "t_coh"]:
                if hw_param == "t_coh":
                    val_range = [2**i for i in range(10, 16)]
                else:
                    max_param_val = 1
                    variants = 10
                    val_range = np.round(np.arange(max_param_val / variants, max_param_val + max_param_val / variants, max_param_val / variants), 2)

                hw_varying_experiment(hw_param, val_range, parameters, min_dists=0, max_dists=1, number_of_swaps=2)


if __name__ == "__main__":
    single_asym_test()