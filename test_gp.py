from collections import OrderedDict
import itertools
import time
import pytest
from asymmetric_sequences import generate_sequences
from gp_utils import catalan_number, generalized_catalan_number, get_asym_protocol_space, get_asym_protocol_space_size, get_protocol_enum_space, get_no_of_permutations_per_swap, get_swap_space


@pytest.mark.parametrize("nodes, kappa", [(5, 0), (5, 1), (5, 2), (6, 1), (7, 0)])
def test_asymmetric_something(nodes, kappa):
    S = nodes - 1
    space = []
    swap_space = get_swap_space(S)
    assert len(swap_space) == catalan_number(nodes-2), f"Expected {catalan_number(nodes-2)} swap sequences, got {len(swap_space)}"
    
    for swap_sequence in swap_space:
        swap_seq = [f"s{i}" for i in swap_sequence]
        joined_sequences = list(generate_sequences(swap_seq=swap_seq, kappa=kappa))
        space.extend([tuple(protocol) for protocol in joined_sequences])

    expected_protocols = generalized_catalan_number(S, kappa+1)
    print(f"\nNumber of protocols for {nodes} nodes and kappa {kappa}: {len(space)}")
    assert len(space) == expected_protocols, f"Expected {expected_protocols} protocols, got {len(space)}"


@pytest.mark.parametrize("nodes, max_dists", [(5, 2), (6, 1), (7, 0)])
def test_space_asym_protocols(nodes, max_dists):
    space = get_asym_protocol_space(nodes, max_dists)
    print(f"\nNumber of protocols for {nodes} nodes and max_dists {max_dists}: {len(space)}")
    expected_protocols = get_asym_protocol_space_size(nodes, max_dists)
    assert len(space) == expected_protocols, \
        f"Expected {get_asym_protocol_space_size(nodes, max_dists)} protocols, got {len(space)}"


@pytest.mark.parametrize("min_dists, max_dists, number_of_swaps", [
    (1, 3, 0),
    (0, 5, 1),
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


if __name__ == "__main__":
    pytest.main(["-sv", "test_gp.py"])