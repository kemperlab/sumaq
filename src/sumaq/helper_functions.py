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
from openfermion import jordan_wigner, FermionOperator


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
    operators: dict[str, list[float]], N_sites: int
) -> tuple[NDArray, list[str]]:
    """
    Creates the Jordan-Wigner transformed fermionic annihilation and creation operators from Openfermion syntax.
    The dictionary `operators` consists of keys which are strings of the form `i` or `i^`, where `i` is an integer between `0` and `N_sites-1`
    and is the index of the creation (`i`) or annihilation (`i^`) operator. Each index (and corresponding `^`, if applicable) is separated by
    spaces. The values of `operators` are the leading coefficient of the corresponding operator key.

    For example, if `N_sites` is `2` and `operators` is `{"0^ 1": 1.0, "1^ 0": 2.0}`, then this function will return:\n

    `coeffs = array([0.  +0.25j, 0.75+0.j  , 0.75+0.j  , 0.  -0.25j])`\n
    `paulis = ['YX', 'YY', 'XX', 'XY']`\n

    If, instead, `N_sites` were `3`, then the returned arrays would be:\n

    `coeffs = array([0.  +0.25j, 0.75+0.j  , 0.75+0.j  , 0.  -0.25j])`\n
    `paulis = ['YX-', 'YY-', 'XX-', 'XY-']`\n



    Parameters:
    -----------
    N_sites : int
        The total number of sites which a fermion can occupy.

    Reurns:
    -    ops : dict(jordan_wigner(FermionOperator()))
            The Jordan-Wigner tranformed operators.
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
