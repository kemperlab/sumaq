"""
Testing script for helper_functions.py
--------------------------------------
This script ensures that the helper functions in helper_functions.py are working as expected.
"""

from sumaq.helper_functions import *
import numpy as np


def test_get_ground_state():
    test_hamiltonian = np.array([[0.5, 1], [4, 0.5]])
    expected_result = (-3.5, np.array([-0.70710678, 0.70710678]))
    result = get_ground_state(test_hamiltonian)
    assert np.isclose(result[0], expected_result[0])
    assert np.allclose(result[1], expected_result[1])


def test_get_overlap_matrix():
    test_basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0.70710678, 0.70710678, 0]])
    expected_result = np.array(
        [[1, 0, 0.70710678], [0, 1, 0.70710678], [0.70710678, 0.70710678, 1]]
    )
    assert np.allclose(get_overlap_matrix(test_basis_vectors), expected_result)


def test_get_fidelity():
    test_vector1 = np.array([1, 0, 0])
    test_vector2 = np.array([0.70710678, 0.70710678, 0])
    expected_result = 0.49999999832196845
    assert np.isclose(get_fidelity(test_vector1, test_vector2), expected_result)


def test_normalize():
    test_data = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([0, 0.25, 0.5, 0.75, 1])
    assert np.allclose(normalize(test_data), expected_result)


def test_generate_paulis_from_fermionic_ops():
    test_operators = {"0^ 1": 1.0, "1^ 0": 1.0, "0^ 0": 3.0}
    N_sites = 2
    expected_coeffs = np.array([0.5 + 0.0j, 0.5 + 0.0j, 1.5 + 0.0j, -1.5 + 0.0j])
    expected_paulis = ["YY", "XX", "--", "Z-"]
    coeffs, paulis = generate_paulis_from_fermionic_ops(test_operators, N_sites)
    assert np.allclose(coeffs, expected_coeffs)
    assert paulis == expected_paulis


def test_get_sparse_from_paulis():
    test_coeffs = np.array([1.0, 2.0, 1.0, -3.0])
    test_paulis = ["-X", "Y-", "XX", "YY"]
    expected_sparse_matrix = bsr_array(
        np.array([
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 - 2.0j, 4.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, -2.0 + 0.0j, 0.0 - 2.0j],
            [0.0 + 2.0j, -2.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 2.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        ]))

    print(get_sparse_from_paulis(test_coeffs, test_paulis) - expected_sparse_matrix)
    assert bsr_allclose(get_sparse_from_paulis(test_coeffs, test_paulis),
                        expected_sparse_matrix)

