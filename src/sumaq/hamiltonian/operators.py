from __future__ import annotations

from abc import abstractmethod, ABC

from numpy.typing import NDArray
from scipy.sparse import coo_array

from ..pauli import PauliSentence
from ..fermionic import FermionicSentence
from ..bosonic import BosonicSentence


class Operator(ABC):
    """
    This is the base class all Hamiltonian subclasses will be derived from.
    """

    @abstractmethod
    def dag(self) -> Operator:
        """
        This method returns the conjugate transpose of the current operator.

        Returns:
            Operator : The adjoint (conjugate transpose) of the current operator.
        """
        ...

    def as_matrix(self) -> NDArray:
        return self.as_spmatrix().todense()

    @abstractmethod
    def as_spmatrix(self) -> coo_array:
        ...

    @abstractmethod
    def as_pauli(self) -> PauliSentence:
        ...

    @abstractmethod
    def as_fermionic(self) -> FermionicSentence:
        ...

    @abstractmethod
    def as_bosonic(self) -> BosonicSentence:
        ...

    @abstractmethod
    def __add__(self, rhs: Operator) -> Operator:
        ...

    @abstractmethod
    def __sub__(self, rhs: Operator) -> Operator:
        ...

    @abstractmethod
    def __mul__(self, rhs: float) -> Operator:
        ...

    @abstractmethod
    def __matmul__(self, rhs: Operator) -> Operator:
        ...
