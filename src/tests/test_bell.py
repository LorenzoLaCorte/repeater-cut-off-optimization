import copy
import numpy as np
import pytest

from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.utils.utility_functions import werner_to_fid

from src.core.states import QuantumState
from src.core.bell.state import BellState
from src.core.werner.state import WernerState

@pytest.fixture
def simulation_params():
    """
    Fixture to provide common simulation parameters.
    Note: No Amplitude Damping or Bit Flip in this test
    """
    return {
        "lambdas": np.array([0.85, 0.05, 0.05, 0.05]),
        "state": BellState([0.85, 0.05, 0.05, 0.05]),
        "p_gen": 0.1,
        "p_swap": 0.25,
        "t_coh": 1000,
        "t_trunc": 100,
        "cutoff": 100,
        "cut_type": "memory_time",
        "depolarizing_rate": 0.1,
        "dephasing_rate": 0.05,
    }

def test_swap(simulation_params):
    """
    Test the swap function of the RepeaterChainEvaluation class.
    Considering a simple protocol (0,)
    """
    lambdas = simulation_params["lambdas"]
    p_gen = simulation_params["p_gen"]
    t_trunc = simulation_params["t_trunc"]
    p_swap, t_coh = simulation_params["p_swap"], simulation_params["t_coh"]
    cutoff, cut_type = simulation_params["cutoff"], simulation_params["cut_type"]

    t_list = np.arange(1, t_trunc)
    pmf = p_gen * (1 - p_gen)**(t_list - 1)
    pmf = np.concatenate((np.array([0.]), pmf))
    pmf1 = np.tile(pmf[:, np.newaxis], 4)
    pmf2 = pmf1.copy()
    lambda_func1 = np.tile(lambdas, (t_trunc, 1))
    lambda_func2 = lambda_func1.copy()

    # swap between two identical links
    repeater = RepeaterChainEvaluation(state_type=BellState)
    pmf_swap, state_out = repeater.swapping(
        parameters=simulation_params, pmf1=pmf1, sf1=lambda_func1, pmf2=pmf2, sf2=lambda_func2, 
        p_swap=p_swap, cutoff=cutoff, t_coh=t_coh, cut_type=cut_type)
        
    print(f"Coverages: {sum(pmf_swap)}")
    assert pmf_swap.shape == pmf1.shape, "PMF output shape mismatch"
    assert state_out.shape == lambda_func1.shape, "State output shape mismatch"

    # Now, repeat the process calling the repeater protocol
    parameters = copy.deepcopy(simulation_params)
    parameters["protocol"] = (0,)

    pmf_swap_protocol, state_out_protocol = repeater.nested_protocol(parameters=parameters)

    assert np.allclose([phiplus[0] for phiplus in pmf_swap], pmf_swap_protocol), "PMF mismatch between direct and protocol methods"
    assert np.allclose(state_out, state_out_protocol), "Lamdas mismatch between direct and protocol methods"
    
    print("Swap test passed.")


def test_distillation(simulation_params):
    """
    Test the distillation function of the RepeaterChainEvaluation class.
    This test checks the output shapes of the distillation process.
    """    
    lambdas = simulation_params["lambdas"]
    p_gen = simulation_params["p_gen"]
    t_trunc = simulation_params["t_trunc"]
    cutoff, cut_type = simulation_params["cutoff"], simulation_params["cut_type"]

    """ t_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] """
    t_list = np.arange(1, t_trunc)

    """ pmf = [0.0, 0.1, 0.09, 0.081, 0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.043046721, 0.0387420489] """
    pmf = p_gen * (1 - p_gen)**(t_list - 1)
    pmf = np.concatenate((np.array([0.]), pmf))
    
    """
        00 = array([0., 0., 0., 0.])
        01 = array([0.1, 0.1, 0.1, 0.1])
        02 = array([0.09, 0.09, 0.09, 0.09])
        03 = array([0.081, 0.081, 0.081, 0.081])
        04 = array([0.0729, 0.0729, 0.0729, 0.0729])
        05 = array([0.06561, 0.06561, 0.06561, 0.06561])
        06 = array([0.059049, 0.059049, 0.059049, 0.059049])
        07 = array([0.0531441, 0.0531441, 0.0531441, 0.0531441])
        08 = array([0.04782969, 0.04782969, 0.04782969, 0.04782969])
        09 = array([0.04304672, 0.04304672, 0.04304672, 0.04304672])
        10 = array([0.03874205, 0.03874205, 0.03874205, 0.03874205])
    """
    pmf1 = np.tile(pmf[:, np.newaxis], 4)
    pmf2 = pmf1.copy()

    """ 00 = array([0.85, 0.05, 0.05, 0.05]) ..."""
    lambda_func1 = np.tile(lambdas, (t_trunc, 1))
    lambda_func2 = lambda_func1.copy()

    # distillation between two identical links
    repeater = RepeaterChainEvaluation(state_type=BellState)

    pmf_dist, state_out = repeater.distillation(
        simulation_params,
        pmf1, lambda_func1, pmf2, lambda_func2, 
        cutoff, simulation_params["t_coh"], cut_type
    )
    
    """ pmf_dist
        00 = array([3.85513738e-10, 3.85513738e-10, 3.85513738e-10, 3.85513738e-10])
        01 = array([0.0082, 0.0082, 0.0082, 0.0082])
        02 = array([0.02141665, 0.02141665, 0.02141665, 0.02141665])
        03 = array([0.03069639, 0.03069639, 0.03069639, 0.03069639])
        04 = array([0.03696796, 0.03696796, 0.03696796, 0.03696796])
        05 = array([0.04095373, 0.04095373, 0.04095373, 0.04095373])
        06 = array([0.04321478, 0.04321478, 0.04321478, 0.04321478])
        07 = array([0.04418608, 0.04418608, 0.04418608, 0.04418608])
        08 = array([0.04420412, 0.04420412, 0.04420412, 0.04420412])
        09 = array([0.0435285, 0.0435285, 0.0435285, 0.0435285])
        10 = array([0.04235888, 0.04235888, 0.04235888, 0.04235888])
    """
    """ w_out
        00 = array([3.04492332e-10, 1.59999272e-10, 2.87428021e-12, 2.87428021e-12])
        01 = array([0.88414633, 0.10365855, 0.00609756, 0.00609756])
        02 = array([0.87102609, 0.11677805, 0.00609793, 0.00609793])
        03 = array([0.86064017, 0.12716337, 0.00609823, 0.00609823])
        04 = array([0.85069294, 0.13710999, 0.00609853, 0.00609853])
        05 = array([0.84096487, 0.14683745, 0.00609884, 0.00609884])
        06 = array([0.8314192 , 0.1563825 , 0.00609915, 0.00609915])
        07 = array([0.82205961, 0.16574146, 0.00609946, 0.00609946])
        ...
        # TODO: if im correct, we are keeping track of 4 pmfs and then only keeping one in the end?
    """
    assert pmf_dist.shape == pmf1.shape, "PMF output shape mismatch"
    assert state_out.shape == lambda_func1.shape, "State output shape mismatch"

    # Now, repeat the process calling the repeater protocol
    parameters = copy.deepcopy(simulation_params)
    parameters["protocol"] = (1,)

    pmf_dist_protocol, state_out_protocol = repeater.nested_protocol(parameters=parameters)

    assert np.allclose([phiplus[0] for phiplus in pmf_dist], pmf_dist_protocol), "PMF mismatch between direct and protocol methods"
    assert np.allclose(state_out, state_out_protocol), "Lamdas mismatch between direct and protocol methods"
    
    print("Distillation test passed.")


def test_bell_vs_werner(simulation_params):
    """
    Test the output shapes of the Bell and Werner states.
    """
    parameters = copy.deepcopy(simulation_params)
    del parameters["cutoff"]
    del parameters["cut_type"]
    del parameters["depolarizing_rate"]
    del parameters["dephasing_rate"]
    parameters["protocol"] = (0,)

    # Bell diagonal protocol
    repeater: RepeaterChainEvaluation = RepeaterChainEvaluation(state_type=BellState)
    pmf_bell, lambda_func = repeater.nested_protocol(parameters=parameters)

    # Werner protocol
    repeater = RepeaterChainEvaluation(state_type=WernerState)
    del parameters["lambdas"]
    del parameters["state"]
    parameters["w0"] = 0.80 # w_0 = 0.8 -> lambda = 0.85
    parameters["state"] = WernerState(w0=parameters["w0"])

    pmf_werner, w_func = repeater.nested_protocol(parameters=parameters)

    assert np.allclose(pmf_werner, pmf_bell), "PMF mismatch between Bell and Werner states"
    fids1 = [l[0] for l in lambda_func]
    fids2 = [werner_to_fid(w) for w in w_func]
    assert np.allclose(pmf_bell, pmf_werner), "PMF mismatch between Bell and Werner states"
    print(f"Fidelities (Bell vs Werner): {[f' {f1:.4f} (B) vs {f2:.4f} (W)' for f1, f2 in zip(fids1[:20], fids2[:20])]} ...")
    assert np.allclose(fids1, fids2), "Fidelity mismatch between Bell and Werner states"
