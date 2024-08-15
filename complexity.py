import logging
import numpy as np
from scipy.special import binom
from argparse import ArgumentParser, Namespace

from distillation_gp_utils import get_protocol_enum_space, get_no_of_permutations_per_swap
import matplotlib.pyplot as plt

logging.getLogger().level = logging.INFO
CHECK_PERCENTAGE = 0.01 # Percentage of times to check the complexity


def analytical_no_dp_complexity(min_dists, max_dists, swaps):
    complexity = 0

    for k in range(min_dists, max_dists + 1):
        complexity += binom((swaps+k), swaps) * (swaps + k)

    return int(complexity)


def analytical_dp_complexity(min_dists, max_dists, swaps):
    rest_complexity = 0

    for k in range(min_dists, max_dists + 1):
        if k == 0:
            rest_complexity += swaps - 1
        else:
            for t in range(0, swaps + 1):
                rest_complexity += binom((k-1)+(swaps-t), k-1) * t

    total_complexity = int(rest_complexity) + get_no_of_permutations_per_swap(min_dists, max_dists, swaps)
    return total_complexity, rest_complexity


def compute_complexity(min_dists, max_dists, min_swaps, max_swaps, dp=False):
    complexity = 0
    total_rest_complexity = 0
    
    for number_of_swaps in range(min_swaps, max_swaps + 1):
        to_check = True if (np.random.rand() < CHECK_PERCENTAGE and max_dists < 10) else False

        if to_check:
            logging.info(f"\n\nChecking complexity for swaps={number_of_swaps}")
            protocol_space = get_protocol_enum_space(min_dists, max_dists, number_of_swaps)

        if not dp:
            round_complexity = analytical_no_dp_complexity(min_dists, max_dists, number_of_swaps)
            complexity += round_complexity

            # Randomly check that analytical complexity is correct
            if to_check:
                exp_complexity = sum([len(protocol) for protocol in protocol_space])
                if exp_complexity != round_complexity:
                    print(f"Expected complexity: {exp_complexity}, Actual complexity: {complexity}")
                    print(f"Protocol space: {protocol_space}")
                    raise AssertionError
                else:
                    logging.info(f"Complexity no_dp verified for swaps={number_of_swaps}")
        
        else:
            round_complexity, rest_complexity = analytical_dp_complexity(min_dists, max_dists, number_of_swaps)
            total_rest_complexity += int(rest_complexity)
            complexity += round_complexity

            # Randomly check that analytical complexity is correct
            if to_check:
                exp_complexity = 0
                cache = {}
                for protocol in protocol_space:
                    for i in range(len(protocol), 0, -1):
                        if protocol[:i] not in cache:
                            exp_complexity += 1
                            cache[protocol[:i]] = True
                if exp_complexity != round_complexity:
                    logging.warn(f"Expected complexity: {exp_complexity}, Actual complexity: {complexity}")
                    logging.warn(f"Protocol space: {protocol_space}")
                    raise AssertionError
                else:
                    logging.info(f"Complexity dp verified for swaps={number_of_swaps}")
    
    if dp:
        return complexity, total_rest_complexity
    else:
        return complexity


def plot_complexity(max_swaps, min_dists, max_dists):
    possible_swaps = range(1, 6, 4)
    possible_max_dists = range(1, max_dists+1)

    complexities_no_dp = {}
    complexities_dp = {}

    logging.info(f"\nComputing complexities...")

    for swaps in possible_swaps:
        for max_dists in possible_max_dists:
            complexities_no_dp[swaps, max_dists] = compute_complexity(min_dists, max_dists, swaps, swaps)
            complexities_dp[swaps, max_dists]    = compute_complexity(min_dists, max_dists, swaps, swaps, dp=True)
            logging.info(f"NO DP \t (swaps={swaps}, max_dists={max_dists}): \t {complexities_no_dp[swaps, max_dists]}")
            logging.info(f"DP    \t (swaps={swaps}, max_dists={max_dists}): \t {complexities_dp[swaps, max_dists]}")
    
    fig, axs = plt.subplots(1, len(possible_swaps), figsize=(8 * len(possible_swaps), 6))

    if np.ndim(axs) == 0:
        axs = np.expand_dims(axs, axis=0)

    for i, swaps in enumerate(possible_swaps):
        ax = axs[i]
        ax.set_title(f"{swaps} {'swaps' if swaps > 1 else 'swap'}, "
                        f"$N = {1+2**swaps}$")
        ax.set_xlabel("Maximum Number of distillations")
        if i == 0:
            ax.set_ylabel("Number of evaluations performed")
        ax.set_xticks(possible_max_dists)

        complexities_dp_total = {key: value[0] for key, value in complexities_dp.items()}
        complexities_dp_rest = {key: value[1] for key, value in complexities_dp.items()}

        for max_dists in possible_max_dists:
            ax.plot(max_dists, complexities_dp_total[swaps, max_dists], color='blue', marker='x', label="With DP, Total" if max_dists == 1 else "")
            ax.plot(max_dists, complexities_dp_rest[swaps, max_dists], color='green', marker='x', label="With DP, Rest" if max_dists == 1 else "")            
            ax.plot(max_dists, complexities_no_dp[swaps, max_dists], color='red', marker='o', label="Without DP" if max_dists == 1 else "")
    ax.legend()

    title = (
        f"Complexity Analysis for Dynamic Programming (DP), "
        f"from {min_dists} to {max_dists} rounds of distillation"
    )   

    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("complexity.png", dpi=300)
    logging.info("\n\nPlot saved!")

    
    logging.info(f"\n\nShowing Evaluations...")

    for key, value in complexities_dp.items():
        logging.info(f"Evaluations with DP for \t(swaps={key[0]}, max_dists={key[1]}):\t{value}")
        logging.info(f"Rest evaluations with DP for \t(swaps={key[0]}, max_dists={key[1]}):\t{value[1]}")


if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--swaps", type=int, default=5, help="Number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=15, help="Maximum amount of distillations to be performed")
      
    args: Namespace = parser.parse_args()

    swaps: int = args.swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists

    logging.info(f"\n\nComplexity without DP: {compute_complexity(min_dists, max_dists, swaps, swaps)}")
    logging.info(f"Complexity with DP: {compute_complexity(min_dists, max_dists, swaps, swaps, dp=True)[0]}\n")

    plot_complexity(swaps, min_dists, max_dists)