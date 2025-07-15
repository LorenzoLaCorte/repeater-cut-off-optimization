import numpy as np
from src.core.states.state import QuantumState

WFunc = np.ndarray

class WernerState(QuantumState):
    """
    Werner state representation.
    """
    def __init__(self, w0: float):
        """
        Initialize a Werner state with a given parameter w0.

        Parameters
        ----------
        w0 : float
            The Werner parameter, which must be a real number between 0 and 1.
        """
        self.w0 = w0
        if isinstance(self.w0, list):
            if not all(np.isreal(w) and 0.0 <= w <= 1.0 for w in self.w0):
                raise InvalidWernerParameterError(f"Invalid Werner parameter w0 = {self.w0}")
        elif not np.isreal(self.w0) or self.w0 < 0.0 or self.w0 > 1.0:
            raise InvalidWernerParameterError(f"Invalid Werner parameter w0 = {self.w0}")
        
        
    def __repr__(self):
        return f"WernerState(w0={self.w0})"
    

    def get_generation_sf(self, t_trunc) -> WFunc:
        """
        Generate the state quality function for the Werner state.

        Parameters
        ----------
        t_trunc : int
            The truncation time for the simulation.

        Returns
        -------
        WFunc
            An initial array of state quality values for each time step.
        """
        return np.array([self.w0] * t_trunc)


class InvalidWernerParameterError(Exception):
    def __init__(self, message="Invalid Werner parameter w0. Must be a real number between 0 and 1."):
        super().__init__(message)


def polish_w_func(w_func: WFunc) -> WFunc:
    """
    Polish the state quality function by erasing unrealistic Werner parameters.
    This can happen when the probability is too small ~1.0e-20.
    """
    w_func = np.where(np.isnan(w_func), 1., w_func)
    w_func[w_func > 1.0] = 1.0
    w_func[w_func < 0.] = 0.
    return w_func