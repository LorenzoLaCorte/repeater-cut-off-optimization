from functools import lru_cache
import json
import logging
import math
import numpy as np

from argparse import ArgumentTypeError

import itertools
import ast

from typing import List, Literal, Tuple, TypedDict

from scipy.special import binom
from scipy.optimize import OptimizeResult
from skopt.space import Categorical
from scipy.stats import norm

from asymmetric_swaps import TreeNode, generate_swap_space, generate_edge_combs
from repeater_types import checkAsymProtocol

logging.basicConfig(level=logging.INFO)

# Define the exception for the CDF coverage
class ThresholdExceededError(Exception):
    """
    This exception is raised when the CDF coverage is below the threshold.
    """
    def __init__(self, message="CDF under threshold count incremented", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info

# Define a type for the parameters
class SimParameters(TypedDict):
    protocol: Tuple[int]
    t_coh: float
    p_gen: float
    p_swap: float
    w0: float

# Define the type for the optimizer
OptimizerType = Literal["bf", "gp"]

# Validation function for the optimizer type
def optimizerType(value: str) -> OptimizerType:
    valid_options = ("gp", "bf")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid optimizer type: {value}. Available options are: {', '.join(valid_options)}")
    return value

# Define the type for space_type
SpaceType = Literal["one_level", "strategy", "enumerate", "centerspace", "asymmetric"]

# Validation function for the space type
def spaceType(value: str) -> SpaceType:
    # TODO: refactor: valid_options = SpaceType.__args__
    valid_options = ("one_level", "strategy", "enumerate", "centerspace", "asymmetric")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid space type: {value}. Available options are: {', '.join(valid_options)}")
    return value


def index_lowercase_alphabet(i):
    """
    Takes in input an integer i and returns the corresponding lowercase letter in the alphabet.
    """
    return chr(i + 97)


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

     note: it is used in plots to remove oscillations in the plot.
    """
    new_w_func = w_func.copy()
    for t in range(len(pmf)):
        if pmf[t] < threshold:
            new_w_func[t] = np.nan
    return new_w_func


def get_all_maxima(ordered_results: List[Tuple[np.float64, Tuple[int]]], min_dists, max_dists) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    Get all the maxima from the results: the secret key rate and the protocol
        for the best secret key rate for each distillation number
    """
    maxima = []
    for number_of_dists in range(min_dists, max_dists + 1):
        evaluations = [result for result in ordered_results if result[1].count(1) == number_of_dists]
        # Skip if there are no evaluations for the number of distillations
        if len(evaluations) == 0:
            continue
        maximum = max(evaluations, key=lambda x: x[0])
        # Do not append if the maximum is zero
        if maximum[0] > 0:
            maxima.append(maximum)
    return maxima


def get_protocol_space_size(space_type: SpaceType, min_dists, max_dists, number_of_swaps):
    """
    Returns the total number of protocols to be tested 
        for a specific space type and number of swaps.
    """
    if space_type == "one_level":
        if min_dists == 0:
            return (max_dists - min_dists) * (number_of_swaps + 1) + 1
        else:
            return (max_dists - min_dists + 1) * (number_of_swaps + 1)
    elif space_type == "enumerate" or space_type == "strategy" or space_type == "centerspace":
        return get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps)


def get_no_of_permutations_per_swap(min_dists, max_dists, s):
    """
    Returns the total number of permutations to be tested for a fixed number of swaps.
    """
    return int(sum([binom(s + d, d) for d in range(min_dists, max_dists + 1)]))


def get_analytical_permutations(min_dists, max_dists, min_swaps, max_swaps):
    """
    Returns the total number of permutations to be tested.
    """
    total_permutations = 0
    for s in range(min_swaps, max_swaps + 1):
        permutations = get_no_of_permutations_per_swap(s, min_dists, max_dists)
        total_permutations += permutations
    return total_permutations


def get_protocol_enum_space(min_dists, max_dists, number_of_swaps, skopt_space=False):
    """
    The permutation space is used to test all the possible combinations of distillations for a fixed number of swaps.
    For each number of distillation tested, we test all the possible permutations of distillations. 
        i.e. for 2 swaps and from 0 to 2 distillations, we test:
        - zero distillations
            (0, 0), 
        - one distillation
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
        - two distillations
            (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)
    """
    space = []
    for num_dists in range(min_dists, max_dists + 1):
        base = [1] * num_dists + [0] * number_of_swaps
        perm_set = set(itertools.permutations(base))
        space.extend(perm_set)
    
    # Order the space: first by ascending length of the protocol, then by the ascending binary representation of the protocol tuple
    space = sorted(space, key=lambda x: (len(x), -int(''.join(map(str, x)), 2)))

    analytical_permutations = get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps)
    assert len(space) == analytical_permutations, f"Expected max. {analytical_permutations} permutations, got {len(space)}"
    
    if skopt_space:
        space = [Categorical([''.join(str(tup)) for tup in space], name='protocol')]
    
    return space


def catalan_number(n):
    """
    Returns the n-th Catalan number.
    """
    return math.comb(2*n, n) // (n + 1)

    
def get_asym_protocol_space_size(nodes, max_dists):
    """
    Returns the space of asymmetric protocols to be tested,
     by analytical derivation.
    """
    if nodes < 2:
        raise ValueError("N should be at least 2")
    S = nodes - 1

    expected = catalan_number(nodes-2) * (max_dists+1)**((2*S-1)-1)
    return expected


def get_asym_protocol_space(nodes: int, max_dists: int):
    """
    Returns the space of asymmetric protocols to be tested.
    For each number of distillation tested, we test all the possible asymmetric protocols.
    """
    space = []
    S = nodes - 1

    swap_space = get_swap_space(S)
    assert len(swap_space) == catalan_number(nodes-2), f"Expected {catalan_number(nodes-2)} swap sequences, got {len(swap_space)}"

    for idx, (swap_tree, swap_sequence) in enumerate(swap_space):
        logging.info(f"{idx+1}/{len(swap_space)} Generating protocols for swap sequence {swap_sequence}...")
        space.extend(get_joined_sequences(S, swap_tree, max_dists))

    expected_protocols = get_asym_protocol_space_size(nodes, max_dists)
    assert len(space) == expected_protocols, \
        f"Expected {get_asym_protocol_space_size(nodes, max_dists)} protocols, got {len(space)}"
    
    space = sorted(space, key=lambda x: (len(x), x))
    return space


def get_joined_sequences(S: int, swap_tree: TreeNode, max_dists: int) -> List[Tuple[str]]:
    """
    Returns the space of edge combinations for a given swap tree and a maximum number of distillations.
    """
    n = 2*S - 1
    space: List[Tuple[str]] = []

    for edgeLabeledShape, finalDist in generate_edge_combs(n, swap_tree, max_dists):
        space.append(edgeLabeledShape.get_sequence() + (f'd{S-1}',)*finalDist)

    return space


def get_swap_shape(seq):
    """
    Returns the shape (evolution of the links) of a given sequence of swaps.
    TODO: replace this function with a score given in the swap_space
    """
    left_segments = {f"{i}," for i in seq}
    right_segments = {f"{i+1}," for i in seq}
    segments = left_segments.union(right_segments)
    shape = set()
    for swap in seq:
        # left_segm is the string in the set that has swap as last character
        # right_segm is the string in the set that has left_segm+1 as first character
        left_segm = next(segm for segm in segments if segm.split(",")[-2] == str(swap))
        left_segm_num = int(left_segm.split(",")[-2])
        right_segm = next((segm for segm in segments if segm.split(",")[0] == str(left_segm_num+1)), None)
        segments.discard(left_segm)
        segments.discard(right_segm)
        new_segment = left_segm + right_segm if right_segm else left_segm
        segments.add(new_segment)
        shape.add(new_segment)
    return shape


@lru_cache(maxsize=None)
def get_swap_space(S):
    """
    Returns the swap space for a given number of nodes.
    """
    if S < 1:
        raise ValueError("S should be at least 1")
    
    swap_space = []
    for tree, seq in generate_swap_space(S):
        shape = get_swap_shape([int(s[1:]) for s in seq])
        mean_length = sum(len(s) for s in shape) / len(shape) if shape else 0
        swap_space.append((mean_length, tree, seq))
    
    # Sort the swap_space by the score of the simmetricity
    swap_space = sorted(swap_space, key=lambda x: x[0], reverse=True)
    logging.info(f"Most symmetric shapes: {[s[2] for s in swap_space[-3:]]}")
    return [(x[1], x[2]) for x in swap_space]


def get_protocol_from_center_spacing_symmetricity(gamma, zeta, nodes, max_dists):
    """
    This function generates a protocol from the symmetricity (gamma) and centering of dists (zeta) parameters.
    TODO: try implementing another version with another parameter kappa, which is the number of distillations per segment.
    Parameters:
    - gamma: Symmetricity of the protocol, between 0 and 1.
    - zeta: Centering of the ro. of distillation, between 0 and 1.
    
    Returns:
    - tuple: the protocol to be tested.

    I. Sample of the sequence of swaps
    II. Sample of the sequences of distillations, before and after each swap

    The protocol is generated as follows:
        I. The set of all the possible sequences of swaps is generated.
        They are ordered by the symmetricity of the shape of the swaps, from the least symmetric to the most symmetric.
        The (gamma*len(swap_space)) sequence is selected;
            if gamma = 0, the least symmetric sequence is selected;
            if gamma = 1, the most symmetric sequence is selected.

        II. For the selected sequence of swaps, the set of all the possible sequences of distillations is generated.
            seq = (D_0, swap_0, D_1, swap_1, D_2, ..., D_S, swap_S-1, D_S)
            where D_0, D_1, ..., D_S are the sequences of distillations.
            The sequences of distillations are generated by considering the swaps performed so far.
        The (zeta*len(joined_sequences)) sequence is selected;
            if zeta = 0, the sequence with no rounds of distillation is selected;
            if zeta = 1, the sequence with the maximum number of rounds of distillation is selected.
    """
    protocol = []

    # Sample the sequence of swaps
    swap_space = get_swap_space(nodes-1)
    selected_idx = round(gamma * (len(swap_space)-1))
    swap_tree, swap_sequence = swap_space[selected_idx][0], swap_space[selected_idx][1]
    logging.debug(f"Selected index: {selected_idx+1}/{len(swap_space)} \
                 \nSelected swap sequence: {swap_sequence}")

    # Generate the joined sequences of distillations
    # Sort the joined sequences by their length
    joined_combs = get_joined_sequences(nodes-1, swap_tree, max_dists)
    joined_combs = sorted(joined_combs, key=lambda x: (len(x), x))
    
    len_joined_combs = len(joined_combs)
    logging.debug(f"Number of joined sequences: {len_joined_combs}")

    # Sample the sequence of distillations
    selected_idx = round(zeta * (len_joined_combs-1))
    logging.debug(f"Selected index: {selected_idx+1}/{len_joined_combs}")
    protocol = tuple(joined_combs[selected_idx])

    checkAsymProtocol(protocol, nodes-1)
    logging.debug(f"Generated protocol: {protocol}")
    return protocol


def sample_outcome(strategy, strategy_weight, protocol, idx, dists_target):
    """
    Returns 0 (swap) or 1 (distillation) based on the input parameters.

    Examples:
        Scenario 1: I miss 3 slots and I have 3 distillations missing
            Thus, I distill with p = 100% 
            (i.e., the first coin is depending on what is missing: 3 distillations / 3 slots)
    
        Scenario 2: I still have 3 slots and I have 1 distillation missing
            First of all, I throw a coin
            With 1/3 probability I distill
            While, the 2/3 remaining are distributed as it follows:
            - if I am dists_first, I distill with a prob. of strategy_weight
            - if I am swaps_first, I swap with a prob. of strategy_weight 
            - if I am alternate, 
                - if the previous was swap, I distill with a prob. of strategy_weight
                - if the previous was distill, I swap with a prob. of strategy_weight
    """
    dists_so_far = protocol.count(1)
    dists_remaining = dists_target - dists_so_far
    slots_remaining = len(protocol) - idx

    # Modeling the impact of the distillations applied so far on the decision
    if dists_remaining == 0:
        return 0
    
    dist_prob = dists_remaining/slots_remaining
    first_coin = np.random.choice([0, 1], p=[1 - dist_prob, dist_prob])
    if first_coin == 1:
        return 1
    else: # Modeling the impact of the strategies on the decision
        if strategy == "dists_first":
            return np.random.choice([0, 1], p=[1 - strategy_weight, strategy_weight])
        
        elif strategy == "swaps_first":
            return np.random.choice([0, 1], p=[strategy_weight, 1 - strategy_weight])
        
        elif strategy == "alternate":
            if idx == 0:
                return np.random.choice([0, 1])
            if protocol[idx-1] == 1:
                return np.random.choice([0, 1], p=[1 - strategy_weight, strategy_weight])
            if protocol[idx-1] == 0:
                return np.random.choice([0, 1], p=[strategy_weight, 1 - strategy_weight])

        else:
            raise ValueError("Invalid strategy")
        

def get_protocol_from_strategy(strategy, strategy_weight, number_of_swaps, number_of_dists):
    """
    Returns the protocol to be tested based on the input parameters.
    """
    protocol = [None] * (number_of_swaps + number_of_dists)

    for idx, _ in enumerate(protocol):
        protocol[idx] = int(sample_outcome(strategy, strategy_weight, protocol, idx, number_of_dists))

    assert protocol.count(1) == number_of_dists, f"Expected {number_of_dists} distillations, got {protocol.count(1)}"
    return tuple(protocol)


def get_protocol_from_center_and_spacing(eta, tau, n, k):
    """
    Generates a protocol for distillation based on a normal distribution.
    The protocol is determined by sampling from the distribution and setting specific
        positions to 1, indicating where distillations should occur.

    Parameters:
    - eta: Center of mass, between -1 and 1.
    - tau: Spacing of the distribution, between 0 and 1.
    - n: Number of SWAPs.
    - k: Number of distillations.
    """
    l = n + k

    # Compute the mean and standard deviation of the normal distribution
    mu = (eta + 1) * (l + 1) / 2 
    sigma = l * tau
    dist = norm(loc=mu, scale=sigma)

    # Compute the PDF for a range of values from 0 to l
    pdf_values = dist.pdf(range(0, l))

    for i, pdf in enumerate(pdf_values):
        logging.debug(f"PDF at index {i}: {pdf}")

    # Sample from the normal distribution n times
    dist_idxs = []
    for i in range(k):
        uniform_sample = np.random.uniform(low=0, high=sum(pdf_values))
        idx = np.searchsorted(np.cumsum(pdf_values), uniform_sample)

        assert (idx >= 0 and idx < l), f"Sampled index {idx} is out of bounds."
        dist_idxs.append(int(idx))

        # Avoid sampling the same value again
        pdf_values[idx] = 0

    dist_idxs.sort()

    # Generate the protocol based on sampled indices
    protocol = [0] * l
    for idx in dist_idxs:
        protocol[idx] = 1

    logging.debug(f"Generated protocol: {tuple(protocol)}")

    assert protocol.count(0) == n, f"Expected {n} swapping, got {protocol.count(0)}"
    assert protocol.count(1) == k, f"Expected {k} distillations, got {protocol.count(1)}"
    return tuple(protocol)


def get_protocol_from_distillations(number_of_swaps, number_of_dists, where_to_distill=None):
    """
    Returns the protocol to be tested based on the input parameters.
    
    Parameters:
    - number_of_swaps (int): The number of swaps to be performed.
    - number_of_distillation (int): The number of distillation to be performed.
    - where_to_distill (int): The level at which to perform distillation.
    
    Returns:
    - tuple: the protocol to be tested.
    """
    distillations = [1] * number_of_dists
    swaps = [0] * number_of_swaps

    if number_of_dists == 0:
        protocol = swaps
    else:
        protocol = swaps[:where_to_distill] + distillations + swaps[where_to_distill:]
    
    return tuple(protocol)


def get_t_trunc(p_gen, p_swap, t_coh, nested_swaps, nested_dists, epsilon=0.01):
    """
    This function is derived from Brand et. al, with an adjustment for distillation.
    TODO: it is a very lossy bound, it should be improved, to get the simulation going faster (mantaining cdf coverage).
    Returns the truncation time based on a lower bound of what is sufficient to reach (1-epsilon) of the simulation cdf.
    """
    t_trunc = int((2/p_swap)**(nested_swaps) * (1/p_gen) * (1/epsilon)) # not considering distillation

    p_dist = 0.5 # in the worst case, p_dist will never go below 0.5
    t_trunc *= (1/p_dist)**(nested_dists) # considering distillation, but very unprecise

    # Introduce a factor to reduce the truncation time, as the previous bound is very lossy
    reduce_factor = 10
    t_trunc = min(max(t_coh, t_trunc//reduce_factor), t_coh * 300)
    return int(t_trunc)


def get_ordered_results(result: OptimizeResult, space_type: SpaceType, number_of_swaps) -> List[Tuple[np.float64, Tuple[int]]]:
    """
    This function adjust the results to be positive and returns an ordered list of (key_rate, protocol)
    """
    result.fun = -result.fun
    result.func_vals = [-val for val in result.func_vals]
    assert len(result.x_iters) == len(result.func_vals)
    
    if space_type == "one_level":
        result_tuples = [(  result.func_vals[i], 
                            get_protocol_from_distillations(
                                number_of_swaps=number_of_swaps,
                                number_of_dists=result.x_iters[i][0], 
                                where_to_distill=result.x_iters[i][1])
                         )  
                        for i in range(len(result.func_vals))]

    elif space_type == "enumerate":
        result_tuples = [(result.func_vals[i], ast.literal_eval(result.x_iters[i][0])) for i in range(len(result.func_vals))]

    elif space_type == "strategy" or space_type == "centerspace" or space_type == "asymmetric":
        result_tuples = [(result.func_vals[i], result.x_iters[i][-1]) for i in range(len(result.func_vals))]

    ordered_results = sorted(result_tuples, key=lambda x: x[0], reverse=True)
    return ordered_results


def write_results(filename, parameters, best_results):
    """
    Write optimization results to file.
    """
    with open(filename, 'a') as file:
        output = (
                    f"\nProtocol parameters:\n"
                    f"{{\n"
                    f"    't_coh': {parameters['t_coh']},\n"
                    f"    'p_gen': {parameters['p_gen']},\n"
                    f"    'p_swap': {parameters['p_swap']},\n"
                    f"    'w0': {parameters['w0']}\n"
                    f"}}\n\n")
            
        for number_of_swaps, (best_protocol, best_score, error_coverage) in best_results.items():
            if best_protocol is None:
                print(f"CDF under threshold for {number_of_swaps} swaps")
                output += (
                        f"CDF under threshold for {number_of_swaps} swaps:\n"
                        f"    CDF coverage: {error_coverage*100}%\n\n"
                    )
            else:
                output += (
                        f"Best configuration for {number_of_swaps} swaps:\n"
                        f"    Best Protocol: {best_protocol}\n"
                        f"    Best secret key rate: {best_score}\n\n"
                    )
            
        file.write(output)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config