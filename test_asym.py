import pytest
import numpy as np
import time
from repeater_algorithm import repeater_sim
from utility_functions import get_mean_waiting_time, get_mean_werner, secret_key_rate, werner_to_fid
from logging_utilities import create_iter_kwargs

protocols = [
    ("s0", "s2", "s1"),
    ("d0", "d1", "d2", "d3", "s0", "s2", "s1"),
    ("s0", "s2", "d1", "d3", "s1"),
    ('d0', 'd1', 'd2', 'd3', 's0', 's2', 'd1', 'd3', 's1'),
    ('d0', 'd0', 'd1', 'd1', 'd2', 'd2', 'd3', 'd3', 's0', 's2', 's1'),
]
benchmarks = [
    (0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 0, 1, 0),
    (1, 1, 0, 0),
]


@pytest.mark.parametrize("protocol, benchmark", zip(protocols, benchmarks))
def test_asym_repeater_sim(protocol, benchmark):
    parameters = {
        "p_gen": 0.02,
        "p_swap": 0.85,
        "w0": 0.95,
        "t_coh": 40000,
        "t_trunc": 400000,
    }

    # Test with heterogeneous protocol
    parameters["protocol"] = protocol
    start_time = time.time()
    pmf1, w_func1 = repeater_sim(parameters)
    elapsed1 = time.time() - start_time
    skr1 = secret_key_rate(pmf1, w_func1)

    # Test with benchmark
    parameters["protocol"] = benchmark
    start_time = time.time()
    pmf2, w_func2 = repeater_sim(parameters)
    elapsed2 = time.time() - start_time
    skr2 = secret_key_rate(pmf2, w_func2)
    
    print(f"\nSKR for protocol {protocol}: {skr1}, for benchmark {benchmark}: {skr2}")
    print(f"Elapsed time for protocol {protocol}: {elapsed1}, for benchmark {benchmark}: {elapsed2}")

    assert skr1 == skr2, "The secret key rate is not the same."
    assert np.array_equal(pmf1, pmf2), "The pmf is not the same."
    assert np.array_equal(w_func1, w_func2), "The w_func is not the same."


if __name__ == "__main__":
    pytest.main(["-sv", "test_asym.py"])