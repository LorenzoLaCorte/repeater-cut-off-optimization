"""
This script is used to test the performance of different distillation strategies
 in an extensive way, by using a Gaussian Process to optimize the number of distillations.
"""

from argparse import ArgumentParser, Namespace

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
from matplotlib.ticker import MaxNLocator

import copy
import numpy as np

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)

from utility_functions import secret_key_rate
from utility_functions import pmf_to_cdf
    

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
    """
    new_w_func = w_func.copy()
    for t in range(len(pmf)):
        if pmf[t] < threshold:
            new_w_func[t] = np.nan
    return new_w_func


def save_plot(fig, axs, row_titles, parameters={}, rate=None, exp_name="protocol.png", legend=False, general_title=None):
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

    if general_title is not None:
        fig.suptitle(general_title)

    if row_titles is not None:
        for ax, row_title in zip(axs[:,-1], row_titles):
            ax.text(1.075, 0.5, row_title, transform=ax.transAxes, ha="left", va="center", rotation=90, fontsize=14)

    right_space = 0.95 if row_titles is not None else 0.975
    plt.tight_layout()
    plt.subplots_adjust(right=right_space, top=(0.75 + axs.shape[0]*0.075), hspace=0.2*axs.shape[0])

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


def get_t_trunc(p_gen, p_swap, w0, t_coh, swaps, dists, where_to_distill):
    """
    TODO: this is very unprecise and should be improved, especially in the distillation case.
    Returns the truncation time based on a lower bound of what is sufficient to reach 99% of the simulation cdf.
    """
    t_trunc = int((2/p_swap)**(swaps) * (1/p_gen)) # not considering distillation
    # t_trunc *= t_coh # maybe required 

    decoherence_factor = np.exp(-t_trunc / t_coh)
    if dists != 0:
        w = w0
        for i in range(dists):
            p_dist = (1+w*w) / 2
            w = (2*w + 4*w*w) / (6*p_dist) * decoherence_factor
            
        t_trunc *= (1/p_dist)**(dists) # considering distillation
        t_trunc *= (1/p_dist)**(where_to_distill) # very rough approximation

    t_trunc = min(max(10000, int(t_trunc)), t_coh * 300)
    return int(t_trunc)


def sim_distillation_strategies(number_of_dists, where_to_distill):
    """
        Fixed parameters:
            - number of swaps
            - hardware paramters

        The function tests the performance of different distillation strategies,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    parameters["t_trunc"] = get_t_trunc(parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"],
                                        number_of_swaps, number_of_dists, where_to_distill)

    protocol = get_protocol(number_of_swaps=number_of_swaps, number_of_dists=number_of_dists, 
                                            where_to_distill=where_to_distill)
    parameters["protocol"] = protocol
    secret_key_rate = get_protocol_rate(parameters)
    print(f"Protocol {protocol},\t r = {secret_key_rate}")
    
    return secret_key_rate



def single_test(): 
    """
    This function is used to test the repeater simulation with a fixed set of parameters.
     You can call this function from command line:
     ```py -c "from distillation_ml_gp import single_test; single_test()"```
    """
    parameters = {
                    "t_coh": 200,
                    "p_gen": 0.999,
                    "p_swap": 0.4,
                    "w0": 0.999,
                    "t_trunc": 200*300
                }
    parameters["protocol"] = (0, 1, 1, 0, 0)
    print(get_protocol_rate(parameters))    


# TODO: add in the process a count of the times cdf < 0.99 and the average percentage reached by simulations
if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=1, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=5, help="Maximum number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=10, help="Maximum amount of distillations to be performed")
    parser.add_argument("--gp_shots", type=int, default=50, help="Number of shots for Gaussian Process")
    parser.add_argument("--gp_initial_points", type=int, default=20, help="Number of initial points for Gaussian Process")
    parser.add_argument("--filename", type=str, default='output.txt', help="Filename for output")
    
    args: Namespace = parser.parse_args()
    
    min_swaps: int = args.min_swaps
    max_swaps: int = args.max_swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists
    gp_shots: int = args.gp_shots
    gp_initial_points: int = args.gp_initial_points
    filename: str = args.filename

    parameters_set = [
        {
            "t_coh": 10**4,
            "p_gen": 0.5,
            "p_swap": 0.5,
            "w0": 0.9
        }
    ]

    with open(filename, 'w') as file:
        file.write(f"From {min_swaps} to {max_swaps} swaps\nFrom {min_dists} to {max_dists} distillations\n")
        file.write(f"Gaussian process with {gp_shots} evaluations and {gp_initial_points} initial points\n")

    for parameters in parameters_set: 
        best_results = {}

        for number_of_swaps in range(min_swaps, max_swaps+1):
            print(f"\n\nNumber of swaps: {number_of_swaps}")
            space = [
                Integer(min_dists, max_dists, name='number_of_dists'), 
                Integer(0, number_of_swaps, name='where_to_distill')
            ]
            
            @use_named_args(space)
            def objective(**params):
                """
                Objective function, consider the whole space of actions,
                    returning a negative secret key rate
                    in order for the optimizer to maximize the function.
                """
                number_of_dists = params['number_of_dists']
                where_to_distill = params['where_to_distill']
                
                secret_key_rate = sim_distillation_strategies(number_of_dists, where_to_distill)
                
                # The gaussian process minimizes the function, so return the negative of the key rate 
                return -secret_key_rate

            result = gp_minimize(objective, space, n_calls=gp_shots, n_initial_points=gp_initial_points, random_state=0)

            # Get the best parameters and score from results 
            best_parameters = result.x
            best_score = -result.fun

            best_results[number_of_swaps] = (best_parameters, best_score)
        
        with open(filename, 'a') as file:
            output = (
                    f"Protocol parameters:\n"
                    f"{{\n"
                    f"    't_coh': {parameters['t_coh']},\n"
                    f"    'p_gen': {parameters['p_gen']},\n"
                    f"    'p_swap': {parameters['p_swap']},\n"
                    f"    'w0': {parameters['w0']}\n"
                    f"}}\n\n")
            
            for number_of_swaps, (best_parameters, best_score) in best_results.items():
                output += (
                    f"Best configuration for {number_of_swaps} swaps:\n"
                    f"    No. of distillations: {best_parameters[0]},\n"
                    f"    After how many swaps is best to distill: {best_parameters[1]}\n"
                    f"    Best Protocol: {get_protocol(number_of_swaps, best_parameters[0], best_parameters[1])}\n"
                    f"    Best secret key rate: {best_score}\n\n"
                )
            
            file.write(output)
