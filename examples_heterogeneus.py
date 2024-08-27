import copy
import itertools
import time
from pprint import pprint
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from distillation_gp_utils import get_protocol_enum_space
from repeater_algorithm import repeater_sim
from utility_functions import pmf_to_cdf, secret_key_rate, get_mean_waiting_time, get_mean_werner, secret_key_rate


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

    if isinstance(parameters["protocol"], tuple) and isinstance(parameters["protocol"][0], tuple):
        title_protocols = ", ".join([f"{protocol}" for protocol in parameters["protocol"]])
    else:
        title_protocols = f"$N = {1+2**(parameters['protocol'].count(0))}, "
    title_params = (
        f"{title_protocols}, "
        f"$p_{{gen}} = {parameters['p_gen']}, "
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
    fig.savefig(f"{exp_name}_{parameters_str}.png", dpi=300)
    

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


def get_t_coh(L0, T1, T2):
    """
    TODO: include T1 in the formulation
    From T1 (relaxation time) and T2 (dephasing time) and the distance of the generated link
        derive t_coh to use for the link
    """
    c = 3e8
    unit_of_time = L0 / c
    t_coh = (1 / (1/T2 - 1/(2*T1))) / unit_of_time
    return int(t_coh)


def plot_heatmap(results, p_gen_val_range, T2_val_range, fixed_p_gen, fixed_T2):
    """
    Plot a heatmap of the secret key rate results for the heterogeneous protocol.
    """
    # Create 2D arrays for p_gen and T2 values
    p_gen_values = np.array(p_gen_val_range)
    T2_values = np.array(T2_val_range)

    # Create a 2D array for the secret key rate (skr) results
    skr_values = np.zeros((len(p_gen_values), len(T2_values)))

    # Populate the skr_values array with the results
    for (p_gen, T2), skr in results.items():
        p_gen_idx = p_gen_val_range.index(p_gen)
        T2_idx = T2_val_range.index(T2)
        skr_values[p_gen_idx, T2_idx] = skr

    fig, axis = plt.subplots(figsize=(10, 8), dpi=200)
    
    # Plot heatmap
    img = axis.imshow(skr_values, aspect='auto', cmap='Blues', 
                      extent=[min(T2_values), max(T2_values), min(p_gen_values), max(p_gen_values)],
                      origin='lower')
    
    # Plot contour line where SKR = 0
    contour = axis.contour(T2_values, p_gen_values, skr_values, levels=[0], colors='red', linewidths=2)
    axis.clabel(contour, fmt={0: 'SKR = 0'}, inline=True, fontsize=10)

    # Highlight the point (fixed_T2, fixed_p_gen)
    axis.plot(fixed_T2, fixed_p_gen, 'x', markersize=10, color='black',
               label=f'daSilva Solution ({fixed_T2:.3e}, {fixed_p_gen:.3e})')
    axis.legend()

    # Add color bar
    cbar = plt.colorbar(img)
    axis.set_title("Secret Key Rate Heatmap")
    axis.set_xlabel(r"$T_{{2}}$ (seconds)")
    axis.set_ylabel(r"$p_{{gen}}$")
    cbar.ax.set_ylabel(r"Secret Key Rate")
    
    fig.tight_layout()
    fig.savefig("dist_het_heatmap.png")
    

def distillation_protocols():
    """
    Start from daSilva21 solution for the hardware parameters and analyze the effect of distillation
    """
    # daSilva21, Table 1
    L0s = [20000, 50000, 60000] # in meters
    fixed_T1_s = 10.23 * 60 * 60 # in seconds
    fixed_T2_s = 22.79e-3 # in seconds

    parameters = {
        "p_gen": [0.0977, 0.0977, 0.0977],
        "w0": [0.9741333333333334, 0.9741333333333334, 0.9741333333333334],
        "p_swap": 0.9414,
    }

    protocol = ((), (), ())
    n_0_protocol_space = get_protocol_enum_space(min_dists=0, max_dists=1, number_of_swaps=0)
    n_1_protocol_space = get_protocol_enum_space(min_dists=0, max_dists=1, number_of_swaps=1)
    
    # Generate all possible combinations of protocols
    n_0_protocol_combinations = list(itertools.product(n_0_protocol_space, repeat=3))
    mixed_protocol_combinations = list(itertools.product(n_0_protocol_space, n_1_protocol_space)) + \
                                    list(itertools.product(n_1_protocol_space, n_0_protocol_space))
    
    protocol_space = n_0_protocol_combinations + mixed_protocol_combinations
    print(len(protocol_space))

    # Evaluate each protocol
    for protocol in protocol_space:
        # Compute correct L0s, t_cohs for the protocol
        # i.e. consider the sub-tuple with the swap (0) and aggregate its values
        # TODO: do it in a more general way
        if len(protocol) != 3:
            if protocol[0].count(0) == 1:
                L0s = [70000, 60000] # in meters
            else:
                L0s = [20000, 110000]
        else:
            L0s = [20000, 50000, 60000]

        t_cohs = [get_t_coh(L0, fixed_T1_s, fixed_T2_s) for L0 in L0s]
        t_trunc = int(min(t_cohs) * 300)
        
        parameters["t_coh"] = t_cohs
        parameters["t_trunc"] = t_trunc
        parameters["protocol"] = protocol
        
        pmf, w_func = repeater_sim(parameters)
        skr = secret_key_rate(pmf, w_func)
        print(f"Protocol: {protocol}, SKR: {skr}")


def heterogeus_protocols():
    """
    Test the repeater heterogenus algorithm against Table1 proposed by daSilva,
        fixing p_swap, w0, (t_trunc)
        varying p_gens 
        varying T1 and T2 and t_cohs accordingly
    """
    # daSilva21, Table 1.
    fixed_p_gen = 0.0977
    fixed_T1_s = 10.23 * 60 * 60 # in seconds
    fixed_T2_s = 22.79e-3 # in seconds
    
    protocol = ((), (), ())
    L0s = [20000, 50000, 60000] # in meters

    w0 = 0.9741333333333334
    p_swap = 0.9414

    steps = 40
    p_gen_val_range = [(fixed_p_gen*2)/steps * i for i in range(1, steps+1)]
    T2_val_range = [(fixed_T2_s*2)/steps * i for i in range(1, steps+1)]

    results = {}
    for p_gen in p_gen_val_range:
        for T2 in T2_val_range:
            t_cohs = [get_t_coh(L0, fixed_T1_s, T2) for L0 in L0s]
            t_trunc = int(min(t_cohs) * 300)
            parameters = {
                "protocol": protocol,
                "p_gen": [p_gen] * len(protocol),
                "w0": [w0] * len(protocol),
                "t_coh": t_cohs,
                "t_trunc": t_trunc,
                "p_swap": p_swap,
            }

            pmf, w_func = repeater_sim(parameters)
            skr = secret_key_rate(pmf, w_func)
            results[(p_gen, T2)] = skr

    pprint(results)
    plot_heatmap(results, p_gen_val_range, T2_val_range, fixed_p_gen, fixed_T2_s)


def test_heterogeneous_protocol():
    """
    Test the heterogeus protocol against the homogeneous protocol.
    """
    parameters = {
        "protocol": ((), (), (), ()),
        "p_gen": [0.1, 0.1, 0.1, 0.1],
        "w0": [0.99, 0.99, 0.99, 0.99],
        "t_coh": [400, 400, 400, 400],
        "t_trunc": 400*30,
        "p_swap": 0.9,
    }
    pmf_het, w_func_het = repeater_sim(parameters)

    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.1,
        "w0": 0.99,
        "t_coh": 400,
        "t_trunc": 400*30,
        "p_swap": 0.9,
    }
    pmf_hom, w_func_hom = repeater_sim(parameters)

    # Compare the two protocols
    for t in range(1, 100):
        assert pmf_het[t] == pmf_hom[t], f"PMF at time {t} is different"
        assert w_func_het[t] == w_func_hom[t], f"Werner parameter at time {t} is different"


def heterogeus_protocol():
    """
    This example is a simple demonstration of the heterogeus protocol.
    """
    # daSilva21, Table 1. Experimentally-derived baseline parameter values and values of the best solution found using our methodology for the
    # use case discussed in section 4.1.1
    parameters = {
        # A protocol here is a tuple of tuples
        # This example is a 4 nodes (ABCD) repeater chain with 3 segments.
        "protocol": ((), (), ()),
        # with heterogenus success probabilities of entanglement generation
        "p_gen": [0.0977, 
                  0.0977, 
                  0.0977],
        # initial Werner parameters for each segment
        "w0": [0.9741333333333334, 
               0.9741333333333334, 
               0.9741333333333334],
        # different memory coherence time for each segment 
        # in the unit of one attempt of elementary link generation
        "t_coh": [340, 
                  136, 
                  113],
        # truncation time for each segment
        "t_trunc": 300*30,
        # and a fixed success probability of entanglement swap
        "p_swap": 0.9414,
    }
   
    # daSilva21, Table E3, approximately
    # parameters = {
    #     "protocol": ((), (), ()),
    #     "p_gen": [0.002588, 
    #               0.0009187, 
    #               0.0009082],
    #     "w0": [0.9577333333333334, 
    #            0.9524, 
    #            0.9522666666666666],
    #     "t_coh": [75, 
    #               40, 
    #               35],
    #     "t_trunc": 75*300,
    #     "p_swap": 0.8459,
    # }

    print("Deterministic algorithm for heterogenus protocol")
    start = time.time()
    # Run the calculation
    pmf, w_func = repeater_sim(parameters)
    end = time.time()

    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    w_func = remove_unstable_werner(pmf, w_func)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_pmf_cdf_werner(pmf, w_func, 75*30, axs, 1, full_werner=False)
    save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, rate=None, 
            exp_name=f"heterogeneus")
    print("Elapse time\n", end-start)
    print()
    aver_w = get_mean_werner(pmf, w_func)
    aver_t = get_mean_waiting_time(pmf)
    print(f"Average Werner parameter: {aver_w}")
    print(f"Average waiting time: {aver_t}")
    print("secret key rate", secret_key_rate(pmf, w_func))
    
if __name__ == "__main__":
    test_heterogeneous_protocol()   
