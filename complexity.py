import numpy as np
from scipy.special import binom
from argparse import ArgumentParser, Namespace

from distillation_gp_utils import get_permutation_space


def approximate_func():
    '''
    Evaluations with DP for         (max_swaps=1, max_dists=1):     5
    Evaluations with DP for         (max_swaps=1, max_dists=2):     10
    Evaluations with DP for         (max_swaps=1, max_dists=3):     16
    Evaluations with DP for         (max_swaps=1, max_dists=4):     23
    Evaluations with DP for         (max_swaps=1, max_dists=5):     31
    Evaluations with DP for         (max_swaps=1, max_dists=6):     40
    Evaluations with DP for         (max_swaps=1, max_dists=7):     50
    Evaluations with DP for         (max_swaps=1, max_dists=8):     61
    Evaluations with DP for         (max_swaps=1, max_dists=9):     73
    Evaluations with DP for         (max_swaps=1, max_dists=10):    86

    Evaluations with DP for         (max_swaps=2, max_dists=1):     13
    Evaluations with DP for         (max_swaps=2, max_dists=2):     28
    Evaluations with DP for         (max_swaps=2, max_dists=3):     49
    Evaluations with DP for         (max_swaps=2, max_dists=4):     77
    Evaluations with DP for         (max_swaps=2, max_dists=5):     113
    Evaluations with DP for         (max_swaps=2, max_dists=6):     158
    Evaluations with DP for         (max_swaps=2, max_dists=7):     213
    Evaluations with DP for         (max_swaps=2, max_dists=8):     279
    Evaluations with DP for         (max_swaps=2, max_dists=9):     357
    Evaluations with DP for         (max_swaps=2, max_dists=10):    448
    '''
    max_dists = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    evaluations = np.array([13, 28, 49, 77, 113, 158, 213, 279, 357, 448])

    coefficients = np.polyfit(max_dists, evaluations, 2)
    poly_function = np.poly1d(coefficients)

    print("Coefficients of the quadratic function: ", coefficients)
    predicted_evaluations = poly_function(max_dists)

    plt.scatter(max_dists, evaluations, color='blue', label='Original data')
    plt.plot(max_dists, predicted_evaluations, color='red', label='Fitted quadratic function')
    plt.xlabel('max_dists')
    plt.ylabel('Evaluations')
    plt.title('Quadratic Fit to Evaluation Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_complexity(min_dists, max_dists, min_swaps, max_swaps, dp=False):
    complexity = 0
    
    for number_of_swaps in range(min_swaps, max_swaps + 1):
        protocol_space = get_permutation_space(min_dists, max_dists, number_of_swaps)

        if not dp:
            complexity += sum([len(protocol) for protocol in protocol_space])
        
        else:
            cache = {}
            for protocol in protocol_space:
                for i in range(len(protocol), 0, -1):
                    if protocol[:i] not in cache:
                        complexity += 1
                        cache[protocol[:i]] = True
            
    return complexity


import matplotlib.pyplot as plt

def plot_complexity():
    min_swaps = 0
    min_dists = 0

    possible_max_swaps = range(1, 3)
    possible_max_dists = range(1, 11)

    complexities_no_dp = {}
    complexities_dp = {}

    for max_swaps in possible_max_swaps:
        for max_dists in possible_max_dists:
            print(f"Computing complexity for (max_swaps={max_swaps}, max_dists={max_dists})")
            complexities_no_dp[max_swaps, max_dists] = compute_complexity(min_dists, max_dists, min_swaps, max_swaps)
            complexities_dp[max_swaps, max_dists]    = compute_complexity(min_dists, max_dists, min_swaps, max_swaps, dp=True)
    
    fig, axs = plt.subplots(1, len(possible_max_swaps), figsize=(8 * len(possible_max_swaps), 6))

    for i, max_swaps in enumerate(possible_max_swaps):
        ax = axs[i]
        ax.set_title(f"{max_swaps} {'swaps' if max_swaps > 1 else 'swap'}, "
                        f"$N = {1+2**max_swaps}$")
        ax.set_xlabel("Number of distillations")
        if i == 0:
            ax.set_ylabel("Number of evaluations performed")
        ax.set_xticks(possible_max_dists)

        for max_dists in possible_max_dists:
            ax.plot(max_dists, complexities_no_dp[max_swaps, max_dists], color='red', marker='o', label="Without DP" if max_dists == 1 else "")
            ax.plot(max_dists, complexities_dp[max_swaps, max_dists], color='blue', marker='x', label="With DP" if max_dists == 1 else "")
        
        ax.legend()

    fig.suptitle("Complexity Analysis for Dynamic Programming (DP)")  # Add a title to the figure
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("complexity.png", dpi=300)
    
    for key, value in complexities_dp.items():
        print(f"Evaluations with DP for \t(max_swaps={key[0]}, max_dists={key[1]}):\t{value}")

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=0, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=3, help="Maximum number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=5, help="Maximum amount of distillations to be performed")
      
    args: Namespace = parser.parse_args()

    min_swaps: int = args.min_swaps
    max_swaps: int = args.max_swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists

    print(f"Complexity without DP: {compute_complexity(min_dists, max_dists, min_swaps, max_swaps)}")
    print(f"Complexity with DP: {compute_complexity(min_dists, max_dists, min_swaps, max_swaps, dp=True)}")