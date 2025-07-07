'''
extract points from results_bf_enumerate/output_bf_enumerate.txt and results_bf_enumerate--dp/output_bf_enumerate--dp.txt
which have the following format:
    swaps=2, max_dists=1, time=0m9.893s
    swaps=2, max_dists=2, time=0m33.899s
    swaps=2, max_dists=3, time=2m51.714s
    ...

Then plot the points from both files in the same plot (one in red and one in blue).    
'''

from matplotlib import pyplot as plt
import numpy as np
import glob
import re

RESULTS_DIR = 'results_dp_complexity'

def parse_multiple_files(file_list):
    """
    Parse multiple files and aggregate the results into a single dictionary.
    """
    complexities = {}
    for filename in file_list:
        with open(filename, 'r') as file:
            for line in file:
                match = re.match(r'swaps=(\d*), max_dists=(\d+), time=(\d+)m([\d.]+)s', line)
                if match:
                    swaps_str = match.group(1)
                    swaps = int(swaps_str) if swaps_str.isdigit() else 0
                    max_dists = int(match.group(2))
                    time = int(match.group(3)) * 60 + float(match.group(4))
                    complexities[swaps, max_dists] = time
    return complexities


def plot_complexity():
    files_no_dp = glob.glob(f'{RESULTS_DIR}/results_bf_enumerate_*/output_bf_enumerate.txt')
    files_dp = glob.glob(f'{RESULTS_DIR}/results_bf_enumerate--dp_*/output_bf_enumerate--dp.txt')

    complexities_no_dp = parse_multiple_files(files_no_dp)
    complexities_dp = parse_multiple_files(files_dp)

    swaps = list(complexities_dp.keys())[0][0]
    possible_max_dists = [k[1] for k in list(complexities_dp.keys())]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    parameters = {
    't_coh': 400,
    'p_gen': 0.5,
    'p_swap': 0.5,
    'w0': 0.933
    }

    title_params = (
        f"$p_{{gen}} = {parameters['p_gen']}, "
        f"p_{{swap}} = {parameters['p_swap']}, "
        f"w_{{0}} = {parameters['w0']},"
        f"t_{{coh}} = {parameters['t_coh']}$"
    )

    ax.set_title(title_params)
    ax.set_xlabel(r"Maximum Number of Distillations $\beta$")
    ax.set_ylabel("Time Taken (s)")
    ax.set_xticks(possible_max_dists)

    for max_dists in possible_max_dists:
        ax.plot(max_dists, complexities_no_dp[swaps, max_dists], color='red', marker='o', label="Without Memoization" if max_dists == 1 else "")
        ax.plot(max_dists, complexities_dp[swaps, max_dists], color='blue', marker='x', label="With Memoization" if max_dists == 1 else "")
    ax.legend()

    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("complexity_experimental.pdf", dpi=300)
    
    print("Time taken with and without Memoization:\n")
    for (key, dp_value), _ in zip(complexities_dp.items(), complexities_no_dp.values()):
        print(f"Time taken with Memoization    for \t(swaps, dists) = {key}:\t{complexities_dp[key]} s")
        print(f"Time taken without Memoization for \t(swaps, dists) = {key}:\t{complexities_no_dp[key]} s\n")
        
if __name__ == '__main__':
    plot_complexity()