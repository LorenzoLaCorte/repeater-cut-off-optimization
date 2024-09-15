from collections import OrderedDict
import itertools
import logging
import time
import pytest
from gp_utils import catalan_number, get_asym_protocol_space, get_asym_protocol_space_size, get_distillation_per_swap, get_joined_sequences, get_protocol_enum_space, get_no_of_permutations_per_swap, get_protocol_from_center_spacing_symmetricity, get_swap_space
from repeater_types import checkAsymProtocol

logging.basicConfig(level=logging.DEBUG)

@pytest.mark.parametrize("nodes, expected_best_protocol", [
    (5, ("s0", "s2", "s1")), 
    (9, ("s0", "s2", "s1", "s4", "s6", "s5", "s3")),
    (11, ('s0', 's2', 's1', 's4', 's6', 's8', 's7', 's5', 's3'))
])
def test_symmetricity_score(nodes, expected_best_protocol):
    """
    Test the symmetricity score of the asymmetric protocol space.
    """
    swap_space = get_swap_space(nodes-1)
    best_protocol = swap_space[0][1]
    assert best_protocol == expected_best_protocol, f"Expected {expected_best_protocol}, got {best_protocol}"


@pytest.mark.parametrize("nodes, max_dists, gamma, kappa, zeta, tau, expected_protocol", [
    (5, 2, 0, 0, 0, 0, ("s0", "s2", "s1")),
    (5, 2, 0.5, 0.5, 0.5, 0.5, ("s0", "s2", "s1")),
    (5, 2, 1, 1, 1, 1, ("s0", "s2", "s1")),
    (7, 1, 0.5, 0.5, 0.5, 0.5, ("s0", "s2", "s1")),
])
def test_protocol_getter(nodes, max_dists, gamma, kappa, zeta, tau):
    """
    Test the protocol getter using a different method.
    TODO: add 
    """

# @pytest.mark.parametrize("nodes, max_dists, gamma, zeta", [
#     (5, 2, 0, 0),
#     (5, 2, 0.5, 0.5),
#     (5, 2, 1, 1),
#     (7, 1, 0.5, 0.5),
# ])
# def test_protocol_getter(nodes, max_dists, gamma, zeta):
#     """
#     Test the protocol getter using a different method.
#     """
#     # Current method
#     start_time = time.time()
#     protocol = get_protocol_from_center_spacing_symmetricity(gamma, zeta, nodes, max_dists)
#     current_method_time = time.time() - start_time
#     logging.debug(f"Current method time: {current_method_time:.6f} seconds")

#     # Inefficient method
#     start_time = time.time()
#     # Sample the sequence of swaps
#     swap_space = get_swap_space(nodes-1)
#     selected_idx = round(gamma * (len(swap_space)-1))
#     swap_tree, _ = swap_space[selected_idx][0], swap_space[selected_idx][1]

#     # Sample the joined sequences    
#     joined_combs = get_joined_sequences(nodes-1, swap_tree, max_dists)
#     # joined_combs = sorted(joined_combs, key=lambda x: (len(x), x))

#     len_joined_combs = len(joined_combs)
#     assert len_joined_combs == get_distillation_per_swap(nodes-1, max_dists), \
#         f"Expected {get_distillation_per_swap(nodes-1, max_dists)} joined sequences, got {len_joined_combs}"

#     logging.debug(f"Number of joined sequences: {len_joined_combs}")
#     selected_idx = round(zeta * (len_joined_combs-1))
#     logging.debug(f"Selected index: {selected_idx+1}/{len_joined_combs}")
#     protocol_cmp = tuple(joined_combs[selected_idx])
#     inefficient_method_time = time.time() - start_time
#     logging.debug(f"Inefficient method time: {inefficient_method_time:.6f} seconds")

#     assert protocol == protocol_cmp, f"Expected {protocol_cmp}, got {protocol}"


@pytest.mark.parametrize("nodes, max_dists", [(4, 3), (5, 0), (5, 2), (6, 1), (7, 0)])
def test_asymmetric_protocol_space(nodes, max_dists):
    """
    Test the asymmetric protocol space, including the swap space.
    """
    S = nodes - 1
    swap_space = get_swap_space(S)
    assert len(swap_space) == catalan_number(nodes-2), f"Expected {catalan_number(nodes-2)} swap sequences, got {len(swap_space)}"
    
    space = get_asym_protocol_space(nodes, max_dists)
    for sequence in space:
        checkAsymProtocol(sequence)
    print(f"All {len(space)} asymmetric protocol sequences are valid.")

    expected_size = get_asym_protocol_space_size(nodes, max_dists)
    logging.info(f"Expected number of protocols: {expected_size}")
    assert len(space) == expected_size, f"Expected {expected_size} asymmetric protocol sequences, got {len(space)}"

  
@pytest.mark.parametrize("min_dists, max_dists, number_of_swaps", [
    (1, 3, 0),
    (0, 5, 1),
    (2, 7, 3),
    (1, 6, 4),
])
def test_symmetric_protocol_space(min_dists, max_dists, number_of_swaps):
    """
    Test the symmetric protocol space, including a different (and slower) implementation, which is used for testing.
    """
    start_time = time.time()
    test_space = get_protocol_enum_space(min_dists, max_dists, number_of_swaps)
    print(f"\nElapsed time: {time.time() - start_time} seconds (getter, i.e. current implementation)")

    start_time = time.time()
    gen_space = OrderedDict()
    for number_of_dists in range(min_dists, max_dists + 1):
        distillations = [1] * number_of_dists
        swaps = [0] * number_of_swaps
        for perm in itertools.permutations(distillations + swaps):
            if perm.count(1) == number_of_dists:
                gen_space[perm] = None
    gen_space = list(gen_space.keys())
    print(f"Elapsed time: {time.time() - start_time} seconds (generator)")

    assert len(test_space) == get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps), "The symmetric protocol space is not the same."
    for s, ss in zip(test_space, gen_space):
        assert s == ss, "The symmetric protocol space is not the same."


if __name__ == "__main__":
    pytest.main(["-sv", "test_gp.py::test_protocol_getter"])