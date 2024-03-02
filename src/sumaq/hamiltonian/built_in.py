from .operators import *
from ..helper_functions import *

# There's a lot of things we need to finish before these classes are ready to go. But hopefully this is a good start.


class Fermi_Hubbard(Operator):
    """
    The pre-defined Hubbard Hamiltonian.
    """

    def __init___(self, t_ij: float, U: float, N_sites: int, mu: float = 0):
        """
        Parameters:
        -----------
        t_ij: float
            The hopping parameter.
        U: float
            The on-site Coulomb interaction parameter.
        N_sites: int
            The number of sites. Each site contains a spin up and spin down fermion.
        mu: float, default=0
            The chemical potential.
        """
        self.N_sites = N_sites
        self.t_ij = t_ij
        self.U = U
        self.mu = mu

        self.fermi_hubbard_dict = {}

        for i in range(0, 2 * self.N_sites, 2):
            for j in range(i, 2 * self.N_sites - 2, 2):
                self.fermi_hubbard_dict[f"{i}^ {j+2}"] = self.t_ij
                self.fermi_hubbard_dict[f"{j+2}^ {i}"] = self.t_ij

                self.fermi_hubbard_dict[f"{i+1}^ {j+3}"] = self.t_ij
                self.fermi_hubbard_dict[f"{j+3}^ {i+1}"] = self.t_ij

        for i in range(self.N_sites):
            self.fermi_hubbard_dict[f"{2*i}^ {2*i} {2*i+1}^ {2*i+1}"] = self.U
            self.fermi_hubbard_dict[f"{i}^ {i}"] = self.mu

    def dag(self) -> Operator:
        """
        This method returns the conjugate transpose of the current operator.

        Returns:
            Operator : The adjoint (conjugate transpose) of the current operator.
        """
        return self

    def as_spmatrix(self) -> bsr_array:
        """
        This method returns the operator as a sparse matrix.

        Returns:
            coo_array : The operator as a sparse matrix.
        """
        sp_fermi_hubbard = get_sparse_from_paulis(
            *generate_paulis_from_fermionic_ops(self.fermi_hubbard_dict, self.N_sites)
        )
        return sp_fermi_hubbard


class Impurity(Operator):
    """
    The pre-defined Hubbard Hamiltonian.
    """

    def __init___(
        self,
        V_bath: float | list[float],
        e_bath: float | list[float],
        U: float,
        N_sites: int,
        mu: float,
        N_bath: int,
        e_imp: float | list[float],
    ):
        """
        Parameters:
        -----------
        V_bath: float | list[float]
            The bath hybridization.
        e_bath: float | list[float]
            The on-site bath energies.
        U: float
            The on-site Coulomb interaction parameter.
        N_sites: int
            The number of sites. Each site contains a spin up and spin down fermion.
        mu: float, default=0
            The chemical potential.
        N_bath: int
            The number of bath sites. Each site contains a spin up and spin down fermion.
        e_imp: float | list[float]
            The impurity energy.
        """
        self.N_sites = N_sites
        
        if isinstance(V_bath, float):
            self.V_bath = np.array([V_bath])
        elif isinstance(V_bath, list):
            self.V_bath = np.array(V_bath)
        
        if isinstance(e_bath, float):
            self.e_bath = np.array([e_bath])
        elif isinstance(e_bath, list):
            self.e_bath = np.array(e_bath)