"""
Helper Functions for the Modules
--------------------------------
This script conains general functions needed to generate objects or
processes within the main modules. Functions that are useful for quick
calculations or data processing are also included here.
"""

import numpy as np
import scipy
from numpy.typing import NDArray
from typing import Literal
from openfermion import jordan_wigner, FermionOperator
from qiskit.quantum_info import SparsePauliOp


def get_ground_state(hamiltonian: NDArray) -> tuple[float, NDArray]:
    """
    Quickly returns the ground state energy and ground state vector of some
    hamiltonian without having to call scipy or numpy and manually select these values.

    Parameters:
    -----------
    hamiltonian : NDArray
        The Hamiltonian with the desired ground state.

    Returns:
    --------
    ground_energy : float
        The energy of the ground state of Hamiltonian.
    ground_vector : NDArray
        The vector of the ground state of Hamiltonian.
    """
    if len(hamiltonian) > 2**10:
        evals, evecs = scipy.sparse.linalg.eigsh(hamiltonian, k=5)
        ground_energy = evals[0]
        ground_vector = evecs[:, 0]
    else:
        evals, evecs = scipy.linalg.eigh(hamiltonian)
        ground_energy = evals[0]
        ground_vector = evecs[:, 0]

    return ground_energy, ground_vector


def get_overlap_matrix(basis_vectors: NDArray) -> NDArray:
    """
    Produces the overlap matrix for a set of basis vectors.

    Parameters:
    -----------
    basis_vectors : NDArray
        The set of basis vectors.

    Returns:
    --------
    overlap : NDArray
        The matrix quantifying the overlap of the basis vectors.
    """
    nvecs = len(basis_vectors)
    overlap = np.zeros([nvecs, nvecs], dtype="complex")

    for i in range(nvecs):
        v_i = basis_vectors[i, :]

        for j in range(i, nvecs):
            v_j = basis_vectors[j, :]

            overlap[i, j] = np.dot(np.conjugate(np.transpose(v_i)), v_j)

            if not i == j:
                overlap[j, i] = np.conjugate(overlap[i, j])

    return overlap


def get_fidelity(vector1: NDArray, vector2: NDArray) -> float:
    """
    Calculates the fidelity between two vectors.
    The fidelity is defined as the squared absolute value of the inner product.

    Parameters:
    -----------
    vector1 : NDArray
        The first vector.
    vector2 : NDArray
        The second vector.

    Returns:
    --------
    fidelity : float
        The fidelity between the two vectors.
    """
    fidelity = np.absolute(np.dot(vector1, vector2)) ** 2
    return fidelity


def generate_paulis_from_fermionic_ops(
    operators: dict[str, float],
    N_sites: int,
) -> tuple[NDArray, list[str]]:
    """
    Creates the Jordan-Wigner transformed fermionic annihilation and creation operators from Openfermion syntax.
    The dictionary `operators` consists of keys which are strings of the form `i` or `i^`, where `i` is an integer between `0` and `N_sites-1`
    and is the index of the creation (`i`) or annihilation (`i^`) operator. Each index (and corresponding `^`, if applicable) should be separated
    by spaces. The values of `operators` are the leading coefficient of the corresponding operator key.

    For example, if `N_sites` is `2` and `operators` is `{"0^ 1": 1.0, "1^ 0": 1.0}`, then this function will return:\n

    `coeffs = array([0.5+0.j, 0.5+0.j])`\n
    `paulis = ['YY', 'XX']`\n

    As another example, if `N_sites` is `2` and `operators` is `{"0^ 1": 1.0, "1^ 0": 2.0}`, then this function will return:\n

    `coeffs = array([0.  +0.25j, 0.75+0.j  , 0.75+0.j  , 0.  -0.25j])`\n
    `paulis = ['YX', 'YY', 'XX', 'XY']`\n

    If, instead, `N_sites` were `3`, then the return output would be:\n

    `coeffs = array([0.  +0.25j, 0.75+0.j  , 0.75+0.j  , 0.  -0.25j])`\n
    `paulis = ['YX-', 'YY-', 'XX-', 'XY-']`\n

    Parameters:
    -----------
    operators : dict(str, list[float])
        The dictionary of fermionic operators. The keys are strings of the form `i` or `i^`, where `i` is an integer between `0` and `N_sites-1`
        and is the index of the creation (`i`) or annihilation (`i^`) operator. Each index (and corresponding `^`, if applicable) should be
        separated by spaces. The values are the leading coefficients of the corresponding operator keys.
    N_sites : int
        The total number of sites which a fermion can occupy.

    Reurns:
    -------
    coeffs : NDArray
        An array of floats corresponding to the coefficients of the Pauli operators.
    paulis : list[str]
        A list of strings corresponding to the Pauli operators.
    """
    jw_fermionic_op_form = jordan_wigner(FermionOperator())

    for el in operators.keys():
        jw_fermionic_op_form += operators[el] * jordan_wigner(FermionOperator(el))

    pauli_data = jw_fermionic_op_form.terms

    coeffs = np.array(list(pauli_data.values()))
    raw_pauli = list(pauli_data.keys())
    paulis = []

    for term in raw_pauli:
        string = "-" * (N_sites - len(term))
        for site in term:
            string = string[: site[0]] + site[1] + string[site[0] :]
        paulis.append(string)

    return coeffs, paulis


def get_sparse_from_paulis(coeffs: NDArray, paulis: list[str]) -> NDArray:
    """
    Converts a list of Pauli operators and corresponding coefficients to a sparse matrix representation.

    Parameters:
    -----------
    coeffs : NDArray
        The coefficients of the Pauli operators.
    paulis : list[str]
        The Pauli operators.

    Returns:
    --------
    sparse_matrix : NDArray
        The sparse matrix representation of the Pauli operators.
    """
    for i, pauli in enumerate(paulis):
        if "-" in pauli:
            paulis[i] = pauli.replace("-", "I")

    sparse_matrix = SparsePauliOp(paulis, coeffs=coeffs).to_matrix()

    return sparse_matrix  # type: ignore


def lehmann_greens_function(
    hamiltonian: NDArray,
    ground_state: NDArray,
    ground_energy: float,
    frequency: NDArray,
    N_sites: int,
    element: tuple[int, int],
    frequency_type: Literal["real", "imaginary"] = "real",
    eta: float = 0.1,
    particle_type: Literal["fermion", "boson"] = "fermion",
) -> NDArray:
    """
    Calculates the Lehmann representation of the Greens function in the frequency domain. Supports both fermionic and bosonic operators and
    real and imaginary frequencies.

    Parameters:
    -----------
    hamiltonian : NDArray
        The Hamiltonian of the system.
    ground_state : NDArray
        The ground state of the system.
    ground_energy : float
        The energy of the ground state of the system.
    frequency : NDArray
        The frequencies at which to calculate the Greens function.
    N_sites : int
        The number of sites in the system.
    element : tuple[int, int]
        The element of the Greens function to calculate.
    frequency_type : str
        The type of frequency to use. Valid options are "imaginary" and "real". Default is "real".
    eta : float
        The small, positive value to use if `frequency_type` is "real". Default is 0.1.
    particle_type : str
        The type of particle to use. Valid options are "fermion" and "boson". Default is "fermion".

    Returns:
    --------
    g_ij : NDArray
        The (i,j)-th element of the Greens function.
    """
    g_ij = np.zeros(len(frequency), dtype=complex)

    coeffs, paulis = generate_paulis_from_fermionic_ops(
        {str(element[0]) + "^": 1.0}, N_sites
    )
    ci_dag = get_sparse_from_paulis(coeffs, paulis)
    ci_dag_psi = ci_dag @ ground_state

    coeffs, paulis = generate_paulis_from_fermionic_ops(
        {str(element[1]) + "^": 1.0}, N_sites
    )
    cj_dag = get_sparse_from_paulis(coeffs, paulis)
    cj_dag_psi = cj_dag @ ground_state

    for i, w in enumerate(frequency):
        if frequency_type == "imag":
            mat1 = (1j * w + ground_energy) * np.eye(2**N_sites) - hamiltonian
            mat2 = (1j * w - ground_energy) * np.eye(2**N_sites) + hamiltonian
        elif frequency_type == "real":
            mat1 = (w + ground_energy + 1j * eta) * np.eye(2**N_sites) - hamiltonian
            mat2 = (w - ground_energy + 1j * eta) * np.eye(2**N_sites) + hamiltonian
        else:
            raise ValueError(
                f'Expected `frequency_type` to be "real" or "imaginary", but found {frequency_type}'
            )

        g_ci_dag_psi = np.linalg.solve(mat1, ci_dag_psi)
        g_cj_dag_psi = np.linalg.solve(mat2, cj_dag_psi)

        g_ij[i] = np.transpose(np.conjugate(cj_dag_psi)) @ g_ci_dag_psi
        if particle_type == "fermion":
            g_ij[i] += np.transpose(np.conjugate(ci_dag_psi)) @ g_cj_dag_psi
        elif particle_type == "boson":
            g_ij[i] -= np.transpose(np.conjugate(ci_dag_psi)) @ g_cj_dag_psi
        else:
            raise ValueError(
                f'Expected `particle_type` to be "fermion" or "boson", but found {particle_type}'
            )

    return g_ij


def retarded_greens_function(
    hamiltonian: NDArray,
    ground_state: NDArray,
    time_values: NDArray,
    N_sites: int,
    element: tuple[int, int],
    particle_type: Literal["fermion", "boson"] = "fermion",
) -> NDArray:
    """
    Calculates the retarded Greens function in the time domain.

    Parameters:
    -----------
    hamiltonian : NDArray
        The Hamiltonian of the system.
    ground_state : NDArray
        The ground state of the system.
    time_values : NDArray
        The time values at which to calculate the Greens function.
    N_sites : int
        The number of sites in the system.
    element : tuple[int, int]
        The element of the Greens function to calculate.
    particle_type : str
        The type of particle to use. Valid options are "fermion" and "boson". Default is "fermion".


    Returns:
    --------
    g_ij : NDArray
        The (i,j)-th element of the Greens function.
    """
    g_ij = np.zeros(len(time_values), dtype=complex)

    coeffs, paulis = generate_paulis_from_fermionic_ops({str(element[0]): 1.0}, N_sites)
    ci = get_sparse_from_paulis(coeffs, paulis)

    coeffs, paulis = generate_paulis_from_fermionic_ops(
        {str(element[0]) + "^": 1.0}, N_sites
    )
    ci_dag = get_sparse_from_paulis(coeffs, paulis)

    coeffs, paulis = generate_paulis_from_fermionic_ops({str(element[1]): 1.0}, N_sites)
    cj = get_sparse_from_paulis(coeffs, paulis)

    coeffs, paulis = generate_paulis_from_fermionic_ops(
        {str(element[1]) + "^": 1.0}, N_sites
    )
    cj_dag = get_sparse_from_paulis(coeffs, paulis)

    state_dag = np.transpose(np.conjugate(ground_state))

    for i, t in enumerate(time_values):
        e_ith = scipy.linalg.expm(-1j * t * hamiltonian)
        e_ithdag = np.transpose(np.conjugate(e_ith))
        g_ij[i] = -1j * (state_dag @ e_ithdag @ cj @ e_ith @ ci_dag @ ground_state)

        if particle_type == "fermion":
            g_ij[i] += -1j * (state_dag @ cj_dag @ e_ithdag @ ci @ e_ith @ ground_state)
        elif particle_type == "boson":
            g_ij[i] -= -1j * (state_dag @ cj_dag @ e_ithdag @ ci @ e_ith @ ground_state)
        else:
            raise ValueError(
                f'Expected `particle_type` to be "fermion" or "boson", but found {particle_type}'
            )

    return g_ij


def save_dict_to_txt(
    dictionary: dict[str, list[float] | NDArray], file_name: str, path: str
) -> None:
    """
    Saves a dictionary to a .txt file in the format:

    `key1`   `value1_1`   `value1_2`  `...`  `value1_M`\n
    `key2`   `value2_1`   `value2_2`  `...`  `value2_K`\n
    `...`      `...`        `...`     `...`    `...`\n
    `keyN`   `valueN_1`   `valueN_2`  `...`  `valueN_L`\n

    Where the values are separated by tabs and each entry is on its own line.
    ".txt" should be included with `file_name`.

    Parameters:
    -----------
    dictionary : dict
        A dictionary containing the data to be saved.
    file_name : str
        The name of the file to be saved, including the ".txt" suffix.
    path : str
        The location in the local system to save the data.
    """
    with open(path + "/" + file_name, "w") as file:
        for key, values in dictionary.items():
            file.write(key + "\t" + "\t".join(str(val) for val in values) + "\n")


def load_txt_to_dict(file_name: str, path: str) -> dict[str, NDArray]:
    """
    Loads a .txt file in the format:

    `key1`   `value1_1`   `value1_2`  `...`  `value1_M`\n
    `key2`   `value2_1`   `value2_2`  `...`  `value2_K`\n
    `...`      `...`        `...`     `...`    `...`\n
    `keyN`   `valueN_1`   `valueN_2`  `...`  `valueN_L`\n

    to a dictionary. The values are separated by tabs and each entry is on its own line.
    ".txt" should be included with `file_name`.

    Parameters:
    -----------
    file_name : str
        The name of the file to be saved, including the ".txt" suffix.
    folder : str
        The relative folder to save to in the main "data" folder.

    Returns:
    --------
    data : dict
        The data in `file_name` as a dictionary.
    """
    data = {}
    with open(path + "/" + file_name, "r") as file:
        for line in file:
            parts = line.split("\t")
            key = parts[0]
            values = np.array([float(part.strip()) for part in parts[1:]])
            data[key] = values

    return data


def pretty_print(vectors: NDArray | list[NDArray]) -> None:
    """
    Prints the basis vector(s) in a human readable format.
    For example, if the basis vectors are [1,0,0,0] and [0,0,1,0], then this function
    will print:

    `----------- Vector 1 -------------`\n
    `00 :  1`\n
    `----------- Vector 2 -------------`\n
    `10 :  1`\n

    Parameters:
    -----------
    basis : NDArray or list[NDArray]
        The basis vector(s).
    """
    if type(vectors[0]) == NDArray:
        for v, vec in enumerate(vectors):
            print(f"----------- Vector {v+1} -------------")
            nbits = int(np.log2(len(vec)))
            formatstr = "{0:>0" + str(nbits) + "b}"
            ix = -1
            for x in vec:
                ix += 1
                if abs(x) < 1e-6:
                    continue
                else:
                    print(formatstr.format(ix), ": ", np.round(x, 4))
    else:
        nbits = int(np.log2(len(vectors)))
        formatstr = "{0:>0" + str(nbits) + "b}"
        ix = -1
        for x in vectors:
            ix += 1
            if abs(x) < 1e-6:
                continue
            else:
                print(formatstr.format(ix), ": ", np.round(x, 4))


def normalize(data: NDArray) -> NDArray:
    """
    Normalizes some array quickly such that the maximum value is 1 and the minimum value is 0.

    Parameters:
    -----------
    data : NDArray
        The array to be normalized.

    Returns:
    --------
    normalized_data : NDArray
        The normalized array.
    """
    normalized_data = (data - min(data)) / (max(data) - min(data))
    return normalized_data
