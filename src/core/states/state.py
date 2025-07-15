from abc import ABC, abstractmethod

class QuantumState(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"

    @abstractmethod
    def get_generation_sf(self, t_trunc: int):
        """
        Get the state fidelity for generation at time t_trunc.
        This method must be implemented by subclasses.
        """
        pass