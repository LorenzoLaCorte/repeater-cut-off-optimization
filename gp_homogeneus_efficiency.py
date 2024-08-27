"""
....
"""

from argparse import ArgumentParser, Namespace
import logging
import time

from gp_utils import OptimizerType, SimParameters, SpaceType, get_protocol_space_size
from repeater_algorithm import RepeaterChainSimulation
from gp_homogeneus import gaussian_optimization, brute_force_optimization
from gp_plots import plot_gp_optimization_efficiency

logging.getLogger().level = logging.INFO

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--swaps", type=int, default=2, help="Number of levels of SWAP to be performed")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum round of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=2, help="Maximum round of distillations to be performed")

    parser.add_argument("--t_coh", type=int, default=60000, help="Coherence time")
    parser.add_argument("--p_gen", type=float, default=0.002, help="Generation success probability")
    parser.add_argument("--p_swap", type=float, default=0.9, help="Swapping probability")
    parser.add_argument("--w0", type=float, default=0.98, help="Werner parameter")

    args: Namespace = parser.parse_args()

    swaps = args.swaps
    min_swaps = max_swaps = swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists

    t_coh = args.t_coh
    p_gen = args.p_gen
    p_swap = args.p_swap
    w0 = args.w0

    parameters: SimParameters = {
        't_coh': t_coh,
        'p_gen': p_gen,
        'p_swap': p_swap,
        'w0': w0,
    }

    simulator = RepeaterChainSimulation(use_cache=False)

    start_time = time.time()
    correct_results = brute_force_optimization(simulator, parameters, "enumerate", min_swaps, max_swaps, min_dists, max_dists, "output.txt", False)
    correct_maximum = correct_results[swaps][0]
    end_time = time.time()
    bf_time = end_time - start_time

    logging.info(f"Bruteforce true maximum: {correct_maximum}, found in time: {bf_time}")

    if correct_maximum == 0:
        logging.warning("Bruteforce true maximum is zero, aborting...")
        exit()

    # GP Results
    times, best_results = {}, {}

    gp_initial_points = 20
    gp_shots_percentages = [.05, .10]

    tries = 1
    spaces = ["strategy", "centerspace"]

    for percentage in gp_shots_percentages:
        logging.info(f"Testing percentage {percentage}...")

        protocol_space_size = get_protocol_space_size("enumerate", min_dists, max_dists, swaps)
        gp_shots = gp_initial_points + 1 + int(protocol_space_size * percentage)

        times[percentage] = {}
        best_results[percentage] = {}
        
        for space in spaces:
            if space == "enumerate" or space == "one_level":
                continue
            logging.info(f"Testing the space '{space}'...")

            best_results[percentage][space] = []
            times[percentage][space] = []
            for t in range(tries):
                logging.info(f"Simulation n.{t}...")
                start_time = time.time()
                results = gaussian_optimization(simulator, parameters, space, 
                                        min_swaps, max_swaps, min_dists, max_dists, 
                                        gp_shots, gp_initial_points, "output.txt", False)
                end_time = time.time()
                best_results[percentage][space].append(results[swaps][0])
                times[percentage][space].append(end_time - start_time)

    correctness = {}
    rel_times = {}
    for space in spaces:
        correctness[space] = [[res/correct_maximum for res in best_results[percentage][space]] 
                              for percentage in gp_shots_percentages]
        rel_times[space] = [[res/bf_time for res in times[percentage][space]] 
                              for percentage in gp_shots_percentages]
        logging.info(f"Space: {space}\nCorrectness: {correctness[space]}\nTimes: {rel_times[space]}")

    plot_info = {
        'tries': tries,
        'swaps': swaps,
        'min_dists': min_dists,
        'max_dists': max_dists,

        'bf_maximum': correct_maximum,
        'bf_time': bf_time,

        'times': {
            'title': 'GP Optimization Time Efficiency',
            'xlabel': 'Percentage of the Space Cardinality Tested',
            'ylabel': 'Time Needed Ratio (GP to BF)',
            'legend_title': 'Space Type',
            'filename': 'dist_seaborn_time.png'
        },
        'correctness': {
            'title': 'GP Optimization Correctness Optimality',
            'xlabel': 'Percentage of the Space Cardinality Tested',
            'ylabel': 'Ratio of the Maximum (GP to BF)',
            'legend_title': 'Space Type',
            'filename': 'dist_seaborn_correctness.png'
        }
    }

    plot_gp_optimization_efficiency(gp_shots_percentages, correctness, parameters, plot_info, 'correctness')
    plot_gp_optimization_efficiency(gp_shots_percentages, rel_times, parameters, plot_info, 'times')