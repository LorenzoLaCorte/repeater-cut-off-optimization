"""
Produce all the plots for the "preliminary" section of the thesis.
"""

import itertools
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from distillation.distillation_one_level import sim_distillation_strategies
from utility_functions import secret_key_rate
from distillation_alternate import DistillationType, config, entanglement_distillation_runner, plot_pmf_cdf_werner, save_plot

plt.style.use('tableau-colorblind10')
include_markers = False
markers = itertools.cycle(['.']*2+['+']*2+['x']*2)

def plots_alternate():
    for parameters in [
        {
        "p_gen": 0.5,
        "p_swap": 0.5,
        "t_trunc": 1000,
        "t_coh": 400,
        "w0": 0.933
        },
        {
        "p_gen": 0.5,
        "p_swap": 0.5,
        "t_trunc": 1000,
        "t_coh": 400,
        "w0": 1.0
        },
    ]:
        zoom = 10
        fig, axs = plt.subplots(1, 2, figsize=(config['figsize']['width'], config['figsize']['height']))

        for dist_type in DistillationType:
            pmf, w_func = entanglement_distillation_runner(dist_type, parameters)
            plot_pmf_cdf_werner(pmf=pmf, w_func=w_func, trunc=(parameters["t_trunc"]//zoom), axs=axs, row=0, 
                                full_werner=False, 
                                label=f"{dist_type.name.upper().replace('_', '-')}, SKR = {secret_key_rate(pmf, w_func):.5f}")
            
        save_plot(fig=fig, axs=axs, row_titles=None, parameters=parameters, 
                    rate=None, exp_name="alternate", legend=True)


def plots_one_level():
    parameters_set = [
        {
            "p_gen": 0.5,
            "p_swap": 0.5,
            "t_trunc": 1000,
            "t_coh": 400,
            "w0": 0.933
        },
    ]
    sim_distillation_strategies(parameters_set)

if __name__ == "__main__":
    plots_alternate()
    plots_one_level()