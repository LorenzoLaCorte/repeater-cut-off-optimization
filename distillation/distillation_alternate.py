"""
This script is used to run simulations to compare different quantum repeater protocols 
and visualize the performance of these protocols under various conditions. 
We are trying to compare strategies for distillation, and in particular to check at which level it is better to distill
"""

import argparse
import copy
import numpy as np
import itertools

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
from matplotlib.ticker import MaxNLocator

include_markers = False
markers = itertools.cycle(['.']*2+['+']*2+['x']*2)

from repeater_algorithm import repeater_sim
from utility_functions import secret_key_rate, pmf_to_cdf
from gp_utils import load_config, remove_unstable_werner

from enum import Enum

config = load_config('config.json')                            

class DistillationType(Enum):
    """
    Enum class for the distillation type.
    """
    DIST_SWAP = (1, 0, 1, 0)
    SWAP_DIST = (0, 1, 0, 1)
    NO_DIST = (0, 0)


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

    title_params = (
        f"$N = {1+2**(parameters['protocol'].count(0))}, "
        f"p_{{gen}} = {parameters['p_gen']}, "
        f"p_{{swap}} = {parameters['p_swap']}, "
        f"w_0 = {parameters['w0']}, "
        f"t_{{coh}} = {parameters['t_coh']}$"
    )

    title = f"{title_params}"
    if rate is not None:
        title += f" - R = {rate}"
    fig.suptitle(title)


    if row_titles is not None:
        for ax, row_title in zip(axs[:,0], row_titles):
            ax.text(-0.2, 0.5, row_title, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=14)

    left_space = 0.1 if row_titles is not None else 0.06
    plt.tight_layout()
    plt.subplots_adjust(left=left_space, top=(0.725 + axs.shape[0]*0.04), hspace=0.2*axs.shape[0])
    
    parameters_str = '_'.join([f"{key}={value}" for key, value in parameters.items() if key != "protocol"])
    fig.savefig(f"{exp_name}_{parameters_str}.png", dpi=config['dpi'])
    

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
        # "PMF": pmf_copy,
        "CDF": cdf,
        "Werner parameter": remove_unstable_werner(pmf_copy, w_func_copy)
    }
    
    plot_axs = axs[row] if axs.ndim == 2 else axs  # handle both 1D and 2D axes arrays
    
    
    for title, data in plot_data.items():
        if include_markers:
            marker = next(markers)
        
        ax = plot_axs[list(plot_data.keys()).index(title)]
        
        if label is not None and title == "Werner parameter":
            ax.plot(np.arange(trunc), data, label=label, marker=(marker if include_markers else None))
        else:
            ax.plot(np.arange(trunc), data, marker=(marker if include_markers else None), markersize=4)
        
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
    parser = argparse.ArgumentParser(description="Run entanglement distillation with specified parameters.")
    parser.add_argument('--t_trunc', type=int, default=1200, help='Truncation time')
    parser.add_argument('--t_coh', type=int, default=120, help='Coherence time')
    parser.add_argument('--p_gen', type=float, default=0.5, help='Generation probability')
    parser.add_argument('--p_swap', type=float, default=0.5, help='Swapping probability')
    parser.add_argument('--w0', type=float, default=0.933, help='Initial Werner state parameter')
    args = parser.parse_args()

    parameters = {
        "p_gen": args.p_gen,
        "p_swap": args.p_swap,
        "t_trunc": args.t_trunc,
        "t_coh": args.t_coh,
        "w0": args.w0
    }
    
    zoom = 10
    fig, axs = plt.subplots(1, 2, figsize=(config['figsize']['width'], config['figsize']['height']))

    for dist_type in DistillationType:
        pmf, w_func = entanglement_distillation_runner(dist_type, parameters)
        plot_pmf_cdf_werner(pmf=pmf, w_func=w_func, trunc=(parameters["t_trunc"]//zoom), axs=axs, row=0, 
                            full_werner=False, 
                            label=f"{dist_type.name.upper().replace('_', '-')}, R = {secret_key_rate(pmf, w_func):.5f}")
        
    save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, 
              rate=None, exp_name="alternate", legend=True)