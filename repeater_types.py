"""

TODO: types and exception defined in gp_utils should be moved here
"""
import re
from typing import Tuple, TypedDict, Union

class SimParameters(TypedDict):
    """
    Type representing a set of parameters for a generic simulation of the algorithm
    """
    protocol: Union[Tuple[int], Tuple[str]]
    t_coh: float
    p_gen: float
    p_swap: float
    w0: float


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


