from scipy.special import binom
from argparse import ArgumentParser, Namespace

def compute_no_of_permutations(min_dists, max_dists, min_swaps, max_swaps):
    compute_no_of_permutations_per_swap = lambda s, min_dists, max_dists: int(sum([binom(s + d, d) for d in range(min_dists, max_dists + 1)]))
    
    total_permutations = 0
    
    for s in range(min_swaps, max_swaps + 1):
        permutations = compute_no_of_permutations_per_swap(s, min_dists, max_dists)
        print(f"Number of permutations for {s} swaps: {permutations}")
        total_permutations += permutations
    return total_permutations


if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--min_swaps", type=int, default=3, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=3, help="Maximum number of swaps")
    parser.add_argument("--min_dists", type=int, default=0, help="Minimum amount of distillations to be performed")
    parser.add_argument("--max_dists", type=int, default=5, help="Maximum amount of distillations to be performed")
      
    args: Namespace = parser.parse_args()

    min_swaps: int = args.min_swaps
    max_swaps: int = args.max_swaps
    min_dists: int = args.min_dists
    max_dists: int = args.max_dists
 
    print(f"Total number of permutations: {compute_no_of_permutations(min_dists, max_dists, min_swaps, max_swaps)}")