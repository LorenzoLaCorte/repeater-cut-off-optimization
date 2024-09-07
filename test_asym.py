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
    """
    Test the repeater_sim function calling it for an asymmetric (but homogeneus) protocol
        both with the algorithm for symmetric protocols [benchmark]
         and the algorithm for asymmetric protocols.
    """
    parameters = {
        "p_gen": 0.00092,
        "p_swap": 0.85,
        "w0": 0.952,
        "t_coh": 1400000,
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


@pytest.mark.parametrize("p_gen, p_swap, w0, t_coh, t_trunc", [
    (0.00092, 0.85, 0.952, 1400000, 400000),
    (0.000015, 0.85, 0.867, 720000, 400000),
 
])
@pytest.mark.parametrize("heterogeneous_protocol, homogeneous_protocol", zip(protocols, benchmarks))
def test_heterogeneus_repeater_sim(p_gen, p_swap, w0, t_coh, t_trunc, heterogeneous_protocol, homogeneous_protocol):
    """
    Test the repeater_sim function calling it for a homogeneous protocol
        both with the algorithm for symmetric (homogeneous) protocols [benchmark]
         and the algorithm for asymmetric (and heterogeneous) protocols.
    """
    # Consider both protocols and define whatever fixed L0
    L0 = 1000

    # Test with benchmark
    parameters = {
        't_coh': t_coh,
        'p_gen': p_gen,
        'p_swap': p_swap,
        'w0': w0,
        "t_trunc": t_trunc
    }
    parameters["protocol"] = homogeneous_protocol

    start_time = time.time()
    pmf1, w_func1 = repeater_sim(parameters)
    elapsed1 = time.time() - start_time
    skr1 = secret_key_rate(pmf1, w_func1)

    segments = sum([1 for step in heterogeneous_protocol if step.startswith("s")]) + 1
    t_coh_seconds = t_coh * L0 / 299792458 # convert coherence time to seconds

    # Test with heterogeneous protocol
    parameters = {
        't_coh': t_coh_seconds,
        'p_gen': [p_gen]*segments,
        'p_swap': p_swap,
        'w0': [w0]*segments,
        "t_trunc": t_trunc,
        "L0": [L0]*segments # derive this in meters
    }
    parameters["protocol"] = heterogeneous_protocol

    start_time = time.time()
    pmf2, w_func2 = repeater_sim(parameters)
    elapsed2 = time.time() - start_time
    skr2 = secret_key_rate(pmf2, w_func2)

    print(f"\nSKR for homogeneous protocol {homogeneous_protocol}: {skr1}, "
        f"for heterogeneous protocol {heterogeneous_protocol}: {skr2}")
    print(f"Elapsed time for homogeneous protocol {homogeneous_protocol}: {elapsed1}, "
        f"for heterogeneous protocol {heterogeneous_protocol}: {elapsed2}")
    
    assert skr1 == skr2, "The secret key rate is not the same."
    assert np.array_equal(pmf1, pmf2), "The pmf is not the same."
    assert np.array_equal(w_func1, w_func2), "The w_func is not the same."


if __name__ == "__main__":
    pytest.main(["-sv", "test_asym.py::test_heterogeneus_repeater_sim"])