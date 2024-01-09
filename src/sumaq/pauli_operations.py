"""
Pauli Operations
----------------
This module contains several useful operations on Pauli strings. The identity along with the three Pauli matrices are
interchangeably referred to as (0, 1, 2, 3) or (-, X, Y, Z).
"""

import numpy as np

RULES = np.array([1, 3, 1, 3])
SIGN_RULES = np.array([[1, 1, 1, 1], [1, 1, 1j, -1j], [1, -1j, 1, 1j], [1, 1j, -1j, 1]])


def product(sigma1: int, sigma2: int) -> int:
    """
    Finds the unsigned product of two Pauli matrices.

    Parameters:
    -----------
    sigma1, sigma2: int
        The two Pauli matrices. The integer can be between 0 and 3.

    Returns:
    --------
    int
        The product of two Pauli matrices. The product of any matrix with itself gives 0. The product of any matrix with
        0 gives the same matrix. The product of any two different non-identity matrices returns the third non-identity
        matrix
    """
    return (sigma1 + sigma2 * RULES[sigma1]) % 4


def signed_product(sigma1: int, sigma2: int) -> tuple[int, complex]:
    r"""
    Finds the signed product of two Pauli matrices.

    Parameters:
    -----------
    sigma1, sigma2: int
        The two Pauli matrices. The integer can be between 0 and 3.

    Returns:
    --------
    int
        The unsigned product of two Pauli matrices.
    complex
        The sign of the product. If two non-identity matrices are multiplied, the sign will be +i if :math:`\sigma_1
        \sigma_2 = c \sigma_3` is a cyclic permutation of (X, Y, Z) and -i if it is not.
    """
    return product(sigma1, sigma2), complex(SIGN_RULES[sigma1, sigma2])


def string_product(
    string1: tuple[int, ...], string2: tuple[int, ...]
) -> tuple[tuple[int, ...], complex, bool]:
    """
    Returns the signed product of two Pauli strings and whether they commute.

    Parameters:
    -----------
    string1, string2: tuple[int, ...]
        The two Pauli strings.

    Returns:
    --------
    result: tuple[int, ...]
        The product of the two strings.
    sign: complex
        The sign of the product.
    bool
       True if the two strings commute and False otherwise.
    """
    # Consistency check
    if len(string1) != len(string2):
        raise Exception(f"Dimension mismatch ({len(string1)} and {len(string2)})")

    string1_array = np.array(string1)
    string2_array = np.array(string2)
    sign: complex = np.prod(SIGN_RULES[string1_array, string2_array])  # type: ignore
    result: tuple[int, ...] = tuple(
        (string1_array + np.multiply(string2_array, RULES[string1_array])) % 4
    )

    if sign.imag == 0:
        return result, sign, True
    else:
        return result, sign, False


# def mut_irr(n: int, x: float = np.pi) -> list[float]:
#     """Returns a list of n mutually irrational numbers. Takes the optional argument x irrational to build the set"""
#     y = x % 1
#     numbers = [y] * n
#     for i in range(1, n):
#         y = (x * y) % 1
#         numbers[i] = y
#
#     return numbers


def strings_to_dict(
    strings: list[tuple[int, ...]] | tuple[int, ...],
    coefficients: list[complex] | complex,
) -> dict[tuple[int, ...], complex]:
    """
    Returns a dictionary for a Pauli sentence with the Pauli strings as keys and their corresponding coefficients as
    values.

    Parameters:
    -----------
    strings: list[tuple[int, ...]] | tuple[int, ...]:
        Can take one Pauli string or a list of Pauli strings.
    coefficients: list[complex] or complex:
        Can take one coefficient or a list of coefficients.

    Returns:
    --------
    dict[tuple[int, ...], complex]:
        The Pauli sentence as a dictionary.
    """
    # Data reformatting
    coefficients_array = np.array([coefficients]).flatten()
    strings_array = np.array([strings])
    if len(strings_array.shape) > 2:
        strings_array = np.squeeze(strings_array, axis=0)
    # Consistency check
    if len(strings_array) != len(coefficients_array):
        raise Exception(
            f"Length mismatch - strings: {len(strings_array)}, coefficients: {len(coefficients_array)}"
        )

    strings_tuple = tuple(map(tuple, strings_array))
    return dict(zip(strings_tuple, coefficients_array))


def full_sum(
    sentence1: dict[tuple[int, ...], complex],
    sentence2: dict[tuple[int, ...], complex],
    tol: float = 0,
) -> dict[tuple[int, ...], complex]:
    """
    Finds the sum of two Pauli sentences.

    Parameters:
    -----------
    sentence1, sentence2: dict[tuple[int, ...], complex]
        The two Pauli sentences.
    tol: float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result: dict[tuple[int, ...], complex]
        The sum of the two Pauli sentences as a dictionary.
    """
    result = sentence1.copy()
    for key in sentence2.keys():
        result[key] = result.get(key, 0) + sentence2[key]
        if abs(result[key]) <= tol:
            result.pop(key)

    return result


def full_product(
    sentence1: dict[tuple[int, ...], complex],
    sentence2: dict[tuple[int, ...], complex],
    tol: float = 0,
) -> dict[tuple[int, ...], complex]:
    """
    Finds the product of two Pauli sentences.

    Parameters:
    -----------
    sentence1, sentence2: dict[tuple[int, ...], complex]
        The two Pauli sentences.
    tol: float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result: dict[tuple[int, ...], complex]
        The product of the two Pauli sentences as a dictionary.
    """
    result: dict[tuple[int, ...], complex] = {}
    for key1 in sentence1.keys():
        for key2 in sentence2.keys():
            string, sign, c = string_product(key1, key2)
            result[string] = (
                result.get(string, 0) + sign * sentence1[key1] * sentence2[key2]
            )
            if abs(result[string]) <= tol:
                result.pop(string)

    return result


def string_exp(string: tuple[int, ...], angle: float) -> dict[tuple[int, ...], complex]:
    r"""
    Finds the exponential of a Pauli string :math:`\mathrm{e}^{\mathrm{i} x P} = \cos{x} + \mathrm{i}P\sin{x}`.

    Parameters:
    -----------
    string: tuple[int, ...]
        The Pauli string to be exponentiated.
    angle: float
        The angle of rotation.

    Returns:
    --------
    result: dict[tuple[int, ...], complex]
        The resulting Pauli sentence as a dictionary.
    """
    result = {}
    if np.cos(angle) != 0:
        result[(0,) * len(string)] = np.cos(angle)
    if np.sin(angle) != 0:
        result[string] = 1j * np.sin(angle)
    return result


def exp_conjugation(
    generators: list[tuple[int, ...]] | tuple[int, ...],
    angles: list[float] | float,
    sentence: dict[tuple[int, ...], complex],
    tol: float = 0,
) -> dict[tuple[int, ...], complex]:
    r"""
    Returns the conjugation of a Pauli sentence :math:`\mathrm{e}^{\mathrm{i} x_{1} P_1} ...
    \mathrm{e}^{\mathrm{i} x_n P_n} X \mathrm{e}^{-\mathrm{i} x_{n} P_n} ... \mathrm{e}^{-\mathrm{i} x_1 P_1}`.

    Parameters:
    -----------
    generators: list[tuple[int, ...]] | tuple[int, ...]
        Can take a one Pauli string or a list of Pauli strings to be exponentiated.
    angles: list[float] | float
        Can take one angle or a list of angles.
    sentence: dict[tuple[int, ...], complex]
        The Pauli sentence to be conjugated.
    tol: float, default=0
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0.

    Returns:
    --------
    result: dict[tuple[int, ...], complex]
        The resulting Pauli sentence as a dictionary.
    """
    # Data reformatting
    angles_array = np.array([angles]).flatten()
    cosine_array = np.cos(2 * angles_array)
    sine_array = np.sin(2 * angles_array)
    generators_array = np.array([generators])
    if len(generators_array.shape) > 2:
        generators_array = np.squeeze(generators_array, axis=0)

    # Consistency check
    if len(generators_array) != len(angles_array):
        raise Exception(
            f"Length mismatch - generators: {len(generators_array)}, angles: {len(angles_array)}"
        )

    result = sentence.copy()
    for i in range(len(angles_array) - 1, -1, -1):
        temp: dict[tuple[int, ...], complex] = {}
        for key in result:
            coefficient = result[key]
            string, sign, c = string_product(
                tuple(generators_array[i]), key
            )  # type: ignore
            # If the ith exponent commutes with string (key) in the Pauli sentence do nothing
            if c:
                temp[key] = temp.get(key, 0) + coefficient
            # If it doesn't commute it necessary anticommutes. Perform the operation exp(2ixP).string
            else:
                temp[key] = temp.get(key, 0) + cosine_array[i] * coefficient
                temp[string] = (
                    temp.get(string, 0) + sign * 1j * sine_array[i] * coefficient
                )

                if abs(temp[string]) <= tol:
                    temp.pop(string)
            if abs(temp[key]) <= tol:
                temp.pop(key)

        result = temp.copy()

    return result


def trace(sentence: dict[tuple[int, ...], complex]) -> float | complex:
    """
    Finds the normalized trace of a Pauli sentence.

    Parameters:
    -----------
    sentence: dict[tuple[int, ...], complex]
        The Pauli sentence.

    Returns:
    --------
    float | complex
        The trace of the Pauli sentence divided by the length of a Pauli string.
    """
    identity = (0,) * len(next(iter(sentence)))
    return sentence.get(identity, 0)
