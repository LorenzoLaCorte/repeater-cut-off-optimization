"""
... % explain the purpose of this example module
"""

import logging
import time

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
    DIST_L0 = (1, 0, 0)
    DIST_L1 = (0, 1, 0)
    DIST_L2 = (0, 0, 1)
    NO_DISTILLATION = (0, 0)

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
            axs[-1].legend()

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
    plt.subplots_adjust(left=left_space, top=(0.75 + axs.shape[0]*0.04), hspace=0.2*axs.shape[0])

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


if __name__ == "__main__":
    parameters = {"p_gen": 0.1, "p_swap": 0.2, "t_trunc": 20000, 
            "t_coh": 200, "w0": 0.999}
    
    zoom = 1
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    for dist_type in DistillationType:
        pmf, w_func = entanglement_distillation_runner(dist_type, parameters)
        plot_pmf_cdf_werner(pmf=pmf, w_func=w_func, trunc=(parameters["t_trunc"]//zoom), axs=axs, row=0, 
                            full_werner=False, label=f"{dist_type.name.lower().replace('_', ' ')}, r = {secret_key_rate(pmf, w_func):.5f}")
        
    save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, 
              rate=None, exp_name="distillation_levels", legend=True)