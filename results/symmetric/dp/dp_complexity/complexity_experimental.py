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

import re

RESULTS_DIR = 'results_dp_complexity'

def parse_file(filename):
    complexities = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'swaps=(\d+), max_dists=(\d+), time=(\d+)m([\d.]+)s', line)
            if match:
                swaps = int(match.group(1))
                max_dists = int(match.group(2))
                time = int(match.group(3)) * 60 + float(match.group(4))
                complexities[swaps, max_dists] = time
    return complexities


def plot_complexity():
    complexities_no_dp = parse_file(f'{RESULTS_DIR}/results_bf_enumerate/output_bf_enumerate.txt')
    complexities_dp = parse_file(f'{RESULTS_DIR}/results_bf_enumerate--dp/output_bf_enumerate--dp.txt')

    swaps = list(complexities_dp.keys())[0][0]
    possible_max_dists = [k[1] for k in list(complexities_dp.keys())]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_title(f"{swaps} {'swaps' if swaps > 1 else 'swap'}, "
                    f"$N = {1+2**swaps}$")
    ax.set_xlabel("Number of distillations")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(possible_max_dists)

    for max_dists in possible_max_dists:
        ax.plot(max_dists, complexities_no_dp[swaps, max_dists], color='red', marker='o', label="Without DP" if max_dists == 1 else "")
        ax.plot(max_dists, complexities_dp[swaps, max_dists], color='blue', marker='x', label="With DP" if max_dists == 1 else "")
    
    ax.legend()

    fig.suptitle("Experimental Time Taken with and without DP")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("complexity_experimental.png", dpi=300)
    
    print("Time taken with and without DP:\n")
    for (key, dp_value), _ in zip(complexities_dp.items(), complexities_no_dp.values()):
        print(f"Time taken with DP    for \t(swaps, dists) = {key}:\t{complexities_dp[key]} s")
        print(f"Time taken without DP for \t(swaps, dists) = {key}:\t{complexities_no_dp[key]} s\n")
        
if __name__ == '__main__':
    plot_complexity()