"""
    This script is used to run experiments on the truncation time to cover 99.7% of the distribution.
"""
import matplotlib.pyplot as plt
import itertools
import itertools

import numpy as np

from repeater_algorithm import repeater_sim

from utility_functions import pmf_to_cdf

'''
Get in input a set of parameters and return the truncation time to cover 99.7% of the distribution
'''
def trunc_experiment(parameters, epsilon=0.01):
    trunc_to_coverage = {}

    swaps = parameters["protocol"].count(0) if isinstance(parameters["protocol"], (tuple)) else 1
    dists = parameters["protocol"].count(1) if isinstance(parameters["protocol"], (tuple)) else 0

    unit = int((2/parameters["p_swap"])**(swaps+dists) * (1/parameters["p_gen"])) // ((100) * (10**dists))
    
    t_trunc = unit

    while True:
        parameters["t_trunc"] = t_trunc

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
    
    if not isinstance(parameters["protocol"], (tuple)):
        filename = f"1swap_0dists_gen{parameters['p_gen']}_swap{parameters['p_swap']}.png"
    else:
        filename = f"{parameters['protocol'].count(0)}swaps_{parameters['protocol'].count(1)}dists_gen{parameters['p_gen']}_swap{parameters['p_swap']}.png"
    plt.savefig(filename)


def compute_analytical_bound(parameters, epsilon=0.01):
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    w0 = parameters["w0"]
    t_coh = parameters["t_coh"]

    swaps = parameters["protocol"].count(0) if isinstance(parameters["protocol"], (tuple)) else 1
    dists = parameters["protocol"].count(1) if isinstance(parameters["protocol"], (tuple)) else 0

    t_trunc = int((2/p_swap)**(swaps) * (1/p_gen) * (1/epsilon)) # not considering distillation

    decoherence_factor = np.exp(-t_trunc / t_coh)
    if dists != 0:
        w = w0
        for i in range(dists):
            p_dist = (1+w*w*decoherence_factor) / 2
            w = (2*w + 4*w*w) / (6*p_dist) * decoherence_factor
            
        t_trunc += (1/p_dist)*(dists) # considering distillation
        # t_trunc *= (1/p_dist)**(where_to_distill) # very rough approximation

    t_trunc = min(max(0, int(t_trunc)), t_coh * 300)
    return int(t_trunc)


def run_trunc_experiments():
    protocols = [(0), (1,0), (1,1,0), (1,1,1,0), (0,0), (1,0,0), (1,1,0,0), (1,1,1,0,0)]
    p_gens = [0.1]
    p_swaps = [0.1]

    all_combinations = itertools.product(protocols, p_gens, p_swaps)
    comparisons = []
    for protocol, p_gen, p_swap in all_combinations:
        parameters = {
            "protocol": protocol,
            "p_gen": p_gen,
            "p_swap": p_swap,
            "w0": 0.99,
            "t_coh": 10000
        }
        analytical_bound = compute_analytical_bound(parameters)
        trunc_to_coverage = trunc_experiment(parameters)

        experimental_trunc = sorted(trunc_to_coverage.keys())[-1]
        comparisons.append((parameters, analytical_bound, experimental_trunc))
        
        plot_trunc_to_coverage(trunc_to_coverage, parameters)
        
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    w0 = parameters["w0"]
    t_coh = parameters["t_coh"]
    print(f"p_gen: {p_gen}, p_swap: {p_swap}, w0: {w0}, t_coh: {t_coh}")
    
    for comparison in comparisons:
        parameters, analytical_bound, experimental_trunc = comparison
        protocol = parameters["protocol"]
        print(f"\nProtocol: {protocol}, "
            f"\n\tAnalytical Bound: {analytical_bound}, Experimental Trunc: {experimental_trunc}")    

if __name__ == "__main__":
    run_trunc_experiments()
