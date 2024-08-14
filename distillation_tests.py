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

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)
from utility_functions import secret_key_rate

from utility_functions import pmf_to_cdf
from matplotlib.ticker import MaxNLocator
import itertools

from enum import Enum


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


def single_test(): 
    """
    This function is used to test the repeater simulation with a fixed set of parameters.
     You can call this function from command line:
     ```py -c "from distillation_ml_gp import single_test; single_test()"```
    """
    parameters = {
        't_coh': 12000,
        'p_gen': 0.1,
        'p_swap': 0.4,
        'w0': 0.98,
        "t_trunc": 12000
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


def distillation_step(pmf, w_func, parameters, unit_kind="dist"):
    """
    Distills the entanglement between two qubits pairs, given the input PMF and Werner parameter arrays.
    Returns the new PMF and Werner parameter arrays after distillation.
    """
    simulator = RepeaterChainSimulation()
    new_pmf, new_w_func = simulator.compute_unit(
        parameters, pmf, w_func, unit_kind="dist")
    return new_pmf, new_w_func


def swapping_step(pmf, w_func, parameters):
    """
    Swaps the entanglement between two qubits pairs, given the input PMF and Werner parameter arrays.
    Returns the new PMF and Werner parameter arrays after swapping.
    """
    simulator = RepeaterChainSimulation()
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
            pmf_gen, w_gen, parameters)
        # And then perform the swapping between A and B
        pmf_swap1, w_swap1 = swapping_step(
            pmf_dist, w_dist, parameters)
    elif distillation_type == DistillationType.SWAPPING_FIRST:
        # Swap the entanglement between A and B
        pmf_swap_nodist, w_swap_nodist = swapping_step(
            pmf_gen, w_gen, parameters)
        # And then perform the distillation between A and B
        pmf_swap1, w_swap1 = distillation_step(
            pmf_swap_nodist, w_swap_nodist, parameters)
    elif distillation_type == DistillationType.NO_DISTILLATION:
        # Perform the swapping between A and B    
        pmf_swap1, w_swap1 = swapping_step(
            pmf_gen, w_gen, parameters)
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


if __name__ == "__main__":
    parameters = {"p_gen": 0.1, "p_swap": 0.5, "t_trunc": 20000, 
                "t_coh": 400, "w0": 0.999}
    
    zoom = 10
    rows = 5
    fig, axs = plt.subplots(rows, 3, figsize=(15, 4*rows), sharex=True)

    for dist_type in DistillationType:
        pmfs, w_funcs = entanglement_distillation_manual(dist_type, parameters, intermediate=True)
        plot_steps(fig, axs, pmfs, w_funcs, (parameters["t_trunc"]//zoom), dist_type.name.lower())

    save_plot(fig, axs, None, parameters, rate=None, exp_name="distillation_steps", legend=True)