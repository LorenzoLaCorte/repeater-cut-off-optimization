"""

TODO: types and exception defined in gp_utils should be moved here
"""
from argparse import ArgumentTypeError
import re
from typing import Tuple, TypedDict, Union, Literal, List

class ThresholdExceededError(Exception):
    """
    This exception is raised when the CDF coverage is below the threshold.
    """
    def __init__(self, message="CDF under threshold count incremented", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info


class SimParameters(TypedDict):
    """
    Type representing a set of parameters for a generic simulation of the algorithm
    """
    protocol: Union[Tuple[int], Tuple[str]]
    t_coh: float
    p_gen: float
    p_swap: float
    w0: float

# Define the type for the optimizer and space_type
OptimizerType = Literal["bf", "gp"]
SpaceType = Literal["one_level", "strategy", "enumerate", "centerspace", "asymmetric"]

def optimizerType(value: str) -> OptimizerType:
    """
    Validates the optimizer type passed in input
    """
    valid_options = ("gp", "bf")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid optimizer type: {value}. Available options are: {', '.join(valid_options)}")
    return value

def spaceType(value: str) -> SpaceType:
    """
    Validates the space type passed in input
    """
    # TODO: refactor: valid_options = SpaceType.__args__
    valid_options = ("one_level", "strategy", "enumerate", "centerspace", "asymmetric")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid space type: {value}. Available options are: {', '.join(valid_options)}")
    return value


def checkProtocolUnit(punit: str) -> bool:
    """
    Checks if a string is a valid protocol unit
        i.e. a string of one char 's' or 'd' and one (arbitrary high) number
        
    """
    return bool(re.match(r'^[sd]\d+$', punit))


def checkAsymProtocol(protocol: Tuple[str], S: int = None) -> Tuple[str]:
    """
    Validates a string passed in input for running an asymmetric protocol
    If the protocol is valid, the string is translated in an instance of the type
    Otherwise, an exception is thrown
    """
    swapped_segments = []
    for punit in protocol:
        operation = punit[0]
        segment = int(punit[1:])
        
        if operation == 's':
            swapped_segments.append(segment)
        
        # Check the protocol doesn't distill a index associated previously with a swapping
        elif operation == 'd':
            assert segment not in swapped_segments, "The protocol is bad formatted."

        assert checkProtocolUnit(punit), "The protocol is bad formatted."
    
    S = max(swapped_segments) + 2 if S is None else S
    
    # Check if the number is between the allowed indexes for segments
    assert all([0 <= s <= S-2 for s in swapped_segments]) and len(swapped_segments) == S-1, "The protocol is bad formatted."

    return S


