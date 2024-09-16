"""
This file contains the demonstration examples,
for the computation of time and Werner parameters
as well as for optimizing cutoff.
Run the example functions will take typically a few minutes.
The three existing examples are:

swap_protocol:
    A nested swap protocol of level 3, with cutoff for each level.

mixed_protocol:
    A mixed protocol with both swap and distillation,
    where the numbers of segments and qubits are not a power of 2.

optimize_cutoff_time:
    Optimization of cutoff for nested swap protocols.

One can run the script directly by opening a console and typing
'python3 examples.py' or open the folder in a Python IDE.
Notice that this file must be kept in the same folder
as the rest of Python scripts.
To choose which example to run, please change the function name
in the last line of this script.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data, load_data)
from utility_functions import remove_unstable_werner, secret_key_rate, pmf_to_cdf
from distillation.distillation_alternate import save_plot, plot_pmf_cdf_werner

from matplotlib.ticker import MaxNLocator
import itertools



def entanglement_generation(p_gen=0.5, t_trunc=20):
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 0-level swap protocol,
        # as we consider only the entanglement generation.
        "protocol": (),
        # success probability of entanglement generation
        "p_gen": p_gen,
        # truncation time for the repeater scheme.
        # It should be increased to cover more time step
        # if the success proability decreases.
        # Commercial hardware can easily handle up to t_trunc=1e5
        "t_trunc": t_trunc,
        "w0": 0.98, # ignore this for the sake of the example
        "p_swap": 0.5  # ignore this as well
    }
    pmf, _ = repeater_sim(parameters)
    return pmf


def plot_entanglement_generation():
    t_trunc = 20
    p_gens = [0.2, 0.5]
    
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2))    

    for i, p_gen in enumerate(p_gens):
        pmf = entanglement_generation(p_gen, t_trunc)
        pmf = pmf[:t_trunc]
        cdf = np.cumsum(pmf)

        axs[i][0].set_title("CDF")
        axs[i][0].plot(np.arange(t_trunc), cdf)
        axs[i][0].set_xlabel("Waiting time $T$")
        axs[i][0].set_ylabel("Probability")
        axs[i][0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i][0].set_ylim(0, 1)

        axs[i][1].set_title("PMF")
        axs[i][1].plot(np.arange(t_trunc), pmf)
        axs[i][1].set_xlabel("Waiting time $T$")
        axs[i][1].set_ylabel("Probability")
        axs[i][1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i][1].set_ylim(0, 1)

    fig.suptitle(f'Entanglement Generation for p_gen: {", ".join([str(p_gen) for p_gen in p_gens])}, t_trunc: {t_trunc}')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    fig.savefig(f"gen_example_{t_trunc}.png")


def entanglement_swap(p_gen=0.5, p_swap=0.5, w_0=0.99, t_trunc=20):
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 3-level swap protocol, without distillation
        # spanning over 9 nodes (i.e. 8 segments)
        "protocol": (0, 0, 0),
        # success probability of entanglement generation
        "p_gen": p_gen,
        # success probability of entanglement swap
        "p_swap": p_swap,
        # truncation time for the repeater scheme.
        # It should be increased to cover more time step
        # if the success proability decreases.
        # Commercial hardware can easily handle up to t_trunc=1e5
        "t_trunc": t_trunc,
        "w0": w_0,
        # the memory coherence time,
        # in the unit of one attempt of elementary link generation.
        "t_coh": 400,
    }
    pmf, w_func = repeater_sim(parameters)
    
    t = 0
    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    while(pmf[t] < 1.0e-17):
        w_func[t] = np.nan
        t += 1

    return pmf, w_func


def entanglement_dist(p_gen=0.5, p_swap=0.5, w_0=0.99, t_trunc=20):
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 2-level swap protocol
        # - 0-level: entanglement generation
        # - 1-level: entanglement distillation of 2 links into a higher quality one
        # - 2,3,4-level: entanglement swapping
        # It spans over 5 nodes (i.e. 4 segments)
        "protocol": (1, 0, 0, 0),
        "p_gen": p_gen,
        "p_swap": p_swap,
        "w0": w_0,
        "t_coh": 400,
        "t_trunc": t_trunc,
    }

    pmf, w_func = repeater_sim(parameters)
    return pmf, w_func


def entanglement_dist_no_decoherence():
    p_gen = 0.5
    p_swap = 0.5
    w_0 = 0.9
    t_trunc = 300

    test_protocols = {
        (): "only generation",
        (1,): "distillation after generation",
        (1, 1): "two distillations after generation",
        (0,): "only swapping",
        (1, 0): "distillation after generation, then swap",
        (0, 1): "swap after generation, then distillation",
    }

    results = []

    for i, protocol in enumerate(test_protocols.keys()):
        parameters = {
            # not setting the coherence time allows for an easy analytical solution
            "protocol": protocol,
            "p_gen": p_gen,
            "p_swap": p_swap,
            "w0": w_0,
            "t_trunc": t_trunc,
         }

        pmf, w_func = repeater_sim(parameters)
        cdf = pmf_to_cdf(pmf)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plot_pmf_cdf_werner(pmf, w_func, t_trunc//20, axs, i, full_werner=False)
        parameters["t_coh"] = "None"
        save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, rate=None, 
                exp_name=f"{test_protocols[protocol]}")
    
        results.append({
            "protocol": protocol,
            "secret_key_rate": secret_key_rate(pmf, w_func),
            "pmf": pmf,
            "cdf": cdf,
            "w_func": w_func
        })

        print(f"""Protocol {protocol}:
              t_1:      PMF: {pmf[1]:.5f}, CDF: {cdf[1]:.5f}, Werner: {w_func[1]:.5f}
              t_2:      PMF: {pmf[2]:.5f}, CDF: {cdf[2]:.5f}, Werner: {w_func[2]:.5f}
              t_3:      PMF: {pmf[3]:.5f}, CDF: {cdf[3]:.5f}, Werner: {w_func[3]:.5f}
              t_10:     PMF: {pmf[10]:.5f}, CDF: {cdf[10]:.5f}, Werner: {w_func[10]:.5f}
              t_50:     PMF: {pmf[50]:.5f}, CDF: {cdf[50]:.5f}, Werner: {w_func[50]:.5f}
        """)

    logging.getLogger().level = logging.EXP
    filename = f"distillation_analytical_test_{protocol}"
    save_data(filename, results)

    return results

def plot_protocol_func(protocol_func=entanglement_swap):
    t_trunc = 30000

    sim_params = [
        {"p_gen": 0.1, "p_swap": 0.1, "w_0": 0.9, "t_trunc": t_trunc},
        {"p_gen": 0.1, "p_swap": 0.25, "w_0": 0.9, "t_trunc": t_trunc},
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 7.2))
    
    handles, labels = [], []
    for i, sim_param in enumerate(sim_params):
        pmf, w_func = protocol_func(**sim_param)

        w_func[0] = np.nan
        threshold = 1.0e-15
        for idx, t in enumerate(pmf):
            if t < threshold:
                w_func[idx] = np.nan

        pmf, w_func = pmf[:t_trunc], w_func[:t_trunc]
        cdf = np.cumsum(pmf)

        plot_configs = [
            ("PMF", pmf, "Probability"),
            ("CDF", cdf, "Probability"),
            ("Werner", w_func, "Werner parameter")
        ]

        for j, (title, data, ylabel) in enumerate(plot_configs):
            axs[j].set_title(title)
            axs[j].plot(np.arange(t_trunc), data, label=title)
            axs[j].set_xlabel("Waiting time $T$")
            axs[j].set_ylabel(ylabel)
            axs[j].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        handles.append(axs[0].lines[i])
        labels.append(f"{sim_param}")


    if "dist" in protocol_func.__name__:
        plt.suptitle(f"Distillation example")
    else:
        plt.suptitle(f"Swap example")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.5)

    fig.legend(handles, labels)
    fig.savefig(f"{protocol_func.__name__}_example_{t_trunc}.png")


def swap_protocol():
    """
    This example is a simplified version of fig.4 from the paper.
    It calculates the waiting time distribution and the Werner parameter
    with the algorithm shown in the paper.
    A Monte Carlo algorithm is used for comparison.
    """
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 3-level swap protocol (without distillation)
        # spanning over 9 nodes (i.e. 8 segments)
        "protocol": (0, 0, 0),
        # success probability of entanglement generation
        "p_gen": 0.1,
        # success probability of entanglement swap
        "p_swap": 0.5,
        # initial Werner parameter
        "w0": 0.98,
        # memory cut-off time
        # "cutoff": (16, 31, 55),
        # the memory coherence time,
        # in the unit of one attempt of elementary link generation.
        "t_coh": 400,
        # truncation time for the repeater scheme.
        # It should be increased to cover more time step
        # if the success proability decreases.
        # Commercial hardware can easily handle up to t_trunc=1e5
        "t_trunc": 3000,
        # the type of cut-off
        "cut_type": "memory_time",
        # the sample size for the MC algorithm
        "sample_size": 1000000,
        }
    # initialize the logging system
    log_init("sim", level=logging.INFO)
    fig, axs = plt.subplots(2, 2)

    # Monte Carlo simulation
    print("Monte Carlo simulation")
    t_sample_list = []
    w_sample_list = []
    start = time.time()
    # Run the MC simulation
    t_samples, w_samples = repeater_mc(parameters)
    t_sample_list.append(t_samples)
    w_sample_list.append(w_samples)
    end = time.time()
    print("Elapse time\n", end-start)
    print()
    plot_mc_simulation(
        [t_sample_list, w_sample_list], axs,
        parameters=parameters, bin_width=1, t_trunc=2000)

    # Algorithm presented in the paper
    print("Deterministic algorithm")
    start = time.time()
    # Run the calculation
    pmf, w_func = repeater_sim(parameters)
    end = time.time()

    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    w_func = remove_unstable_werner(pmf, w_func)

    print("Elapse time\n", end-start)
    print()
    plot_algorithm(pmf, w_func, axs, t_trunc=2000)
    print("secret key rate", secret_key_rate(pmf, w_func))

    # plot
    legend = None
    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
    fig.savefig("swap_protocol.png")


def mixed_protocol(cutoff=False):
    """
    Here we show a mixed protocol with the number of qubits and segments not
    a power of 2. Notice that it is only for demonstration purpose
    and the protocol is not optimal.
    Setup:
        A four nodes (ABCD) repeater chain with three segments.
        A and D as end nodes each has 3 qubits;
        B and C as repeater nodes each has 6 qubits.
    The name of entangled pairs following the convention:
    span<N>_dist<d>,
    where N is the number of segments this entanglement spans
    and d is the number of elementary links used in the distillation.
    E.g. an elementary link has the name "span1_dist1", while
    the distilled state of two elementary links the name "span1_dist2".
    """
    parameters = {
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.85,
        "t_coh": 800,
        "t_trunc": 3000,
    }
    p_gen = parameters["p_gen"]
    t_trunc = parameters["t_trunc"]
    w0 = parameters["w0"]

    simulator = RepeaterChainSimulation()

    rows = 4
    fig, axs = plt.subplots(rows, 3, figsize=(15, 4*rows), sharex=True)

    ############################
    # Part1
    # Generate entanglement link for all qubits pairs between AB, BC and CD
    pmf_span1_dist1 = np.concatenate(
        (np.array([0.]),  # generation needs at least one step.
        p_gen * (1 - p_gen)**(np.arange(1, t_trunc) - 1))
        )
    w_span1_dist1 = w0 * np.ones(t_trunc)  # initial werner parameter

    ############################
    # Part2: Between A and B, we distill the entanglement twice.
    # We first distill A1-B1 and A2-B2, save the result in A1-B1
    pmf_span1_dist2, w_span1_dist2 = simulator.compute_unit(
        parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="dist")

    # We then distill A1-B1 and A3-B3, obtain a single link A-B
    pmf_span1_dist3, w_span1_dist3 = simulator.compute_unit(
        parameters, pmf_span1_dist2, w_span1_dist2,
        pmf_span1_dist1, w_span1_dist1, unit_kind="dist")
    
    cdf_span1_dist3 = pmf_to_cdf(pmf_span1_dist3)

    axs[0][0].plot(np.arange(t_trunc), pmf_span1_dist3)
    axs[0][0].set_title("PMF")
    axs[0][1].plot(np.arange(t_trunc), cdf_span1_dist3)
    axs[0][1].set_title("CDF")
    axs[0][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span1_dist3, w_span1_dist3))
    axs[0][2].set_title("Werner parameter")
    axs[0][2].set_ylim([0, 1])

    ############################
    # Part3: Among B, C and D. Performed simultaneously as part2
    # We first connect all elementary links between B-C and C-D, and then distill.
    # We begin from swap between B-C and C-D, for all 3 pairs of elementary link.
    pmf_span2_dist1, w_span2_dist1 = simulator.compute_unit(
        parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="swap")

    cdf_span2_dist1 = pmf_to_cdf(pmf_span2_dist1)
    
    axs[1][0].plot(np.arange(t_trunc), pmf_span2_dist1)
    axs[1][1].plot(np.arange(t_trunc), cdf_span2_dist1)
    axs[1][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span2_dist1, w_span2_dist1))
    axs[1][2].set_ylim([0, 1])

    print(f"Werner parameter at time 500 before distillation: {w_span2_dist1[500]}") # DEBUG

    # When B1-D1, B2-D2 and prepared, we distill them
    pmf_span2_dist2, w_span2_dist2 = simulator.compute_unit(
        parameters, pmf_span2_dist1, w_span2_dist1, unit_kind="dist")

    # When B3-D3 is ready, we merge it too with distillation to obtain
    # a single link between B and D
    # Here we add a cutoff on the memory storage time to increase the fidelity,
    # at the cost of a longer waiting time.
    pmf_span2_dist3, w_span2_dist3 = simulator.compute_unit(
        parameters, pmf_span2_dist2, w_span2_dist2,
        pmf_span2_dist1, w_span2_dist1, unit_kind="dist")

    cdf_span2_dist3 = pmf_to_cdf(pmf_span2_dist3)
    axs[2][0].plot(np.arange(t_trunc), pmf_span2_dist3)
    axs[2][1].plot(np.arange(t_trunc), cdf_span2_dist3)
    axs[2][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span2_dist3, w_span2_dist3), label='Without cutoff')
    axs[2][2].set_ylim([0, 1])

    print(f"Werner parameter at time 500 after distillation: {w_span2_dist3[500]}") # DEBUG

    if cutoff:
        parameters["cutoff"] = 50
        pmf_span2_dist3_cut, w_span2_dist3_cut = simulator.compute_unit(
            parameters, pmf_span2_dist2, w_span2_dist2,
            pmf_span2_dist1, w_span2_dist1, unit_kind="dist")
        del parameters["cutoff"]

        cdf_span2_dist3_cut = pmf_to_cdf(pmf_span2_dist3_cut)
        axs[2][0].plot(np.arange(t_trunc), pmf_span2_dist3_cut)
        axs[2][1].plot(np.arange(t_trunc), cdf_span2_dist3_cut)
        axs[2][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span2_dist3_cut, w_span2_dist3_cut), label='With cutoff')

    ############################
    # Part4
    # We connect A-B and B-D with a swap
    pmf_span3_dist3, w_span3_dist3 = simulator.compute_unit(
        parameters, pmf_span1_dist3, w_span1_dist3,
        pmf_span2_dist3, w_span2_dist3, unit_kind="swap")

    cdf_span3_dist3 = pmf_to_cdf(pmf_span3_dist3)
    axs[3][0].plot(np.arange(t_trunc), pmf_span3_dist3)
    axs[3][1].plot(np.arange(t_trunc), cdf_span3_dist3)
    axs[3][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span3_dist3, w_span3_dist3), label='Without cutoff')
    axs[3][2].set_ylim([0, 1])

    if cutoff:
        parameters["cutoff"] = 50
        pmf_span3_dist3_cut, w_span3_dist3_cut = simulator.compute_unit(
            parameters, pmf_span1_dist3, w_span1_dist3,
            pmf_span2_dist3_cut, w_span2_dist3_cut, unit_kind="swap")
        del parameters["cutoff"]

        cdf_span3_dist3_cut = pmf_to_cdf(pmf_span3_dist3_cut)
        axs[3][0].plot(np.arange(t_trunc), pmf_span3_dist3_cut)
        axs[3][1].plot(np.arange(t_trunc), cdf_span3_dist3_cut)
        axs[3][2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span3_dist3_cut, w_span3_dist3_cut), label='With cutoff')
        axs[3][2].set_ylim([0, 1])

    print("secret key rate", secret_key_rate(pmf_span3_dist3, w_span3_dist3))
    if cutoff:
        print("secret key rate with cutoffs", secret_key_rate(pmf_span3_dist3_cut, w_span3_dist3_cut))

    formatted_parameters = ', '.join([f"{key}={value}" for key, value in parameters.items()])

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.94, hspace=0.1)

    if cutoff:
        axs[2][2].legend()
        axs[3][2].legend()
    
    row_titles = ["(a)", "(b)", "(c)", "(d)"]
    for ax, row_title in zip(axs[:,0], row_titles):
        ax.text(-0.2, 0.5, row_title, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=14)

    fig.suptitle(f"Mixed protocol with {formatted_parameters}")
    fig.savefig(f"mixed_protocol_t_trunc{t_trunc}.png")

    # Plot with only final results
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    axs2[0].plot(np.arange(t_trunc), pmf_span3_dist3)
    axs2[0].set_title("PMF")
    axs2[1].plot(np.arange(t_trunc), cdf_span3_dist3)
    axs2[1].set_title("CDF")
    axs2[2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span3_dist3, w_span3_dist3), label='Without cutoff')
    axs2[2].set_title("Werner parameter")
    axs2[2].set_ylim([0, 1])

    if cutoff:
        axs2[0].plot(np.arange(t_trunc), pmf_span3_dist3_cut)
        axs2[1].plot(np.arange(t_trunc), cdf_span3_dist3_cut)
        axs2[2].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span3_dist3_cut, w_span3_dist3_cut), label='With cutoff')
        axs2[2].legend()

    fig2.suptitle(f"Mixed protocol with {formatted_parameters}")
    plt.tight_layout()
    fig2.savefig(f"mixed_protocol_final_t_trunc{t_trunc}.png")


def mixed_protocol_comp():
    """

    """
    t_cohs = [200, 400, 800]

    parameters = {
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.9999,
        "t_trunc": 3000,
    }
    formatted_parameters = ', '.join([f"{key}={value}" for key, value in parameters.items()])

    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    t_trunc = parameters["t_trunc"]
    w0 = parameters["w0"]

    simulator = RepeaterChainSimulation()

    colors = ['blue', 'red']
    rows = 3
    fig, axs = plt.subplots(rows, 1, figsize=(15, 4*rows), sharex=True)

    for index, t_coh in enumerate(t_cohs):
        parameters["t_coh"] = t_coh

        ############################
        # Part1
        # Generate entanglement link for all qubits pairs between AB, BC and CD
        pmf_span1_dist1 = np.concatenate(
            (np.array([0.]),  # generation needs at least one step.
            p_gen * (1 - p_gen)**(np.arange(1, t_trunc) - 1))
            )
        w_span1_dist1 = w0 * np.ones(t_trunc)  # initial werner parameter

        ############################
        # Part2: Between A and B, we distill the entanglement twice.
        # We first distill A1-B1 and A2-B2, save the result in A1-B1
        pmf_span1_dist2, w_span1_dist2 = simulator.compute_unit(
            parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="dist")

        # We then distill A1-B1 and A3-B3, obtain a single link A-B
        pmf_span1_dist3, w_span1_dist3 = simulator.compute_unit(
            parameters, pmf_span1_dist2, w_span1_dist2,
            pmf_span1_dist1, w_span1_dist1, unit_kind="dist")

        ############################
        # Part3: Among B, C and D. Performed simultaneously as part2
        # We first connect all elementary links between B-C and C-D, and then distill.
        # We begin from swap between B-C and C-D, for all 3 pairs of elementary link.
        pmf_span2_dist1, w_span2_dist1 = simulator.compute_unit(
            parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="swap")
        
        axs[index].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span2_dist1, w_span2_dist1), color=colors[0], label=f'Before distillation')

        # When B1-D1, B2-D2 and prepared, we distill them
        pmf_span2_dist2, w_span2_dist2 = simulator.compute_unit(
            parameters, pmf_span2_dist1, w_span2_dist1, unit_kind="dist")

        # When B3-D3 is ready, we merge it too with distillation to obtain
        # a single link between B and D
        # Here we add a cutoff on the memory storage time to increase the fidelity,
        # at the cost of a longer waiting time.
        pmf_span2_dist3, w_span2_dist3 = simulator.compute_unit(
            parameters, pmf_span2_dist2, w_span2_dist2,
            pmf_span2_dist1, w_span2_dist1, unit_kind="dist")

        axs[index].plot(np.arange(t_trunc), remove_unstable_werner(pmf_span2_dist3, w_span2_dist3), color=colors[1], label=f'After distillation')

    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.94, hspace=0.1)

    row_titles = ["t_coh = 400", "t_coh = 800", "t_coh = 1600"]
    for ax, row_title in zip(axs, row_titles):
        ax.text(-0.05, 0.5, row_title, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=14)

    fig.suptitle(f"Mixed protocol, distillation comparison, with {formatted_parameters}")
    fig.savefig(f"m_dist_cmp_pgen{p_gen}_pswap{p_swap}_w{w0}_.png")


def optimize_cutoff_time():
    """
    This example includes the optimization of the memory storage cut-off time.
    Without cut-off, this parameters give zero secret rate.
    With the optimized cut-off, the secret key rate can be increased to
    more than 3*10^(-3).
    Depending on the hardware, running the whole example may take a few hours.
    The uniform cut-off optimization is smaller.
    """
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.98,
        "t_coh": 400,
        "t_trunc": 3000,
        "cut_type": "memory_time",
        }
    log_init("opt", level=logging.INFO)

    # Uniform cut-off optimization. ~ 1-2 min
    logging.info("Uniform cut-off optimization\n")
    # Define optimizer parameters
    opt = CutoffOptimizer(opt_kind="uniform_de", adaptive=True)
    # Run optimization
    best_cutoff_dict = opt.run(parameters)
    # Calculate the secret key rate
    parameters["cutoff"] = best_cutoff_dict["memory_time"]
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.6f}".format(key_rate))
    del parameters["cutoff"]

    # Nonuniform cut-off optimization.  ~ 5 min
    logging.info("Nonuniform cut-off optimization\n")
    opt = CutoffOptimizer(opt_kind="nonuniform_de", adaptive=True)
    best_cutoff_dict = opt.run(parameters)
    parameters["cutoff"] = best_cutoff_dict["memory_time"]
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.6f}".format(key_rate))

    logging.info("No cut-off\n")
    parameters["cutoff"] = np.iinfo(int).max
    pmf, w_func = repeater_sim(parameters=parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate without cut-off: {:.6f}".format(key_rate))
    logging.info("Rate without truncation time: {}\n".format(key_rate))


def no_trunc_swap():
    parameters = {
        "protocol": (),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.98,
        "t_coh": 400,
        "t_trunc": None,
        "cut_type": "memory_time",
    }

    pmf, w_func = repeater_sim(parameters)
    t = 0
    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    while(pmf[t] < 1.0e-17):
        w_func[t] = np.nan
        t += 1

    t_trunc = len(pmf)
    fig, axs = plt.subplots(2, 2)
    plot_algorithm(pmf, w_func, axs, t_trunc)
    print("secret key rate", secret_key_rate(pmf, w_func))

    # plot
    legend = None
    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
    fig.savefig("swap_protocol.png")


'''
Get in input a set of parameters and return the truncation time to cover 99.7% of the distribution
'''
def trunc_experiment(parameters, epsilon=0.01):
    trunc_to_coverage = {}
    
    levels = len(parameters["protocol"]) if isinstance(parameters["protocol"], (tuple)) else 1
    unit = int((2/parameters["p_swap"])**(levels) / parameters["p_gen"])
    
    t_trunc = unit
    while True:
        parameters["t_trunc"] = t_trunc
        parameters["t_coh"] = t_trunc/100
        pmf, _ = repeater_sim(parameters)

        coverage = pmf_to_cdf(pmf)[-1]
        t_trunc += unit

        trunc_to_coverage[t_trunc] = coverage

        if coverage > (1-epsilon):
            return trunc_to_coverage 

def plot_trunc_to_coverage(trunc_to_coverage, parameters):
    plt.figure(figsize=(10, 6))

    lists = sorted(trunc_to_coverage.items())
    x, y = zip(*lists)
    plt.plot(x, y, marker='o')
    plt.title('Time to Coverage')
    plt.xlabel('Truncation Time')
    plt.ylabel('Coverage')
    
    filename = f"{len(parameters['protocol'])}level_gen{parameters['p_gen']}_swap{parameters['p_swap']}.png"
    plt.savefig(filename)


def compute_analytical_bound(parameters, epsilon=0.01):
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    levels = len(parameters["protocol"]) if isinstance(parameters["protocol"], (tuple)) else 1

    bound = int(((2/p_swap)**(levels)) * (1 / p_gen) * (1 / epsilon))
    return bound


def run_trunc_experiments():
    protocols = [()]
    p_gens = [0.1, 0.01]
    p_swaps = [0.1, 0.01]

    all_combinations = itertools.product(protocols, p_gens, p_swaps)
    comparisons = []
    for protocol, p_gen, p_swap in all_combinations:
        parameters = {
            "protocol": protocol,
            "p_gen": p_gen,
            "p_swap": p_swap,
            "w0": 0.98,
        }
        analytical_bound = compute_analytical_bound(parameters)
        trunc_to_coverage = trunc_experiment(parameters)

        experimental_trunc = sorted(trunc_to_coverage.keys())[-1]
        comparisons.append((parameters, analytical_bound, experimental_trunc))
        
        plot_trunc_to_coverage(trunc_to_coverage, parameters)

    logging.getLogger().level = logging.EXP
    filename = f"{len(parameters['protocol'])}level_gen{parameters['p_gen']}_swap{parameters['p_swap']}"
    save_data(filename, comparisons)

if __name__ == "__main__":
    # swap_protocol()
    # mixed_protocol()
    # mixed_protocol_comp()
    # optimize_cutoff_time()
    # plot_protocol_func(protocol_func=entanglement_swap)
    # plot_protocol_func(protocol_func=entanglement_dist)
    # run_trunc_experiments()
    entanglement_dist_no_decoherence()
        