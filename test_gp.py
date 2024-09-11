from collections import OrderedDict
import itertools
import logging
import time
import pytest
from gp_utils import catalan_number, get_asym_protocol_space, get_asym_protocol_space_size, get_protocol_enum_space, get_no_of_permutations_per_swap, get_swap_space
from repeater_types import checkAsymProtocol

logging.basicConfig(level=logging.DEBUG)


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
    pytest.main(["-sv", "test_gp.py"])