from collections import OrderedDict
import itertools
import time
import pytest
from gp_utils import get_protocol_enum_space, get_no_of_permutations_per_swap


@pytest.mark.parametrize("min_dists, max_dists, number_of_swaps", [
    (1, 3, 0),
    (0, 5, 1),
    (2, 9, 2),
    (2, 7, 3),
    (1, 6, 4),
])
def test_get_protocol_enum_space(min_dists, max_dists, number_of_swaps):

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

    assert len(test_space) == get_no_of_permutations_per_swap(min_dists, max_dists, number_of_swaps), "The protocol enum space is not the same."
    for s, ss in zip(test_space, gen_space):
        assert s == ss, "The protocol enum space is not the same."