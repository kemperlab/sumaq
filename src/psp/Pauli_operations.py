# Python version 3.11.5
# Created on December 15, 2023

"""
Pauli Operations
----------------
This module contains several useful operations on Pauli strings. The identity along with the three Pauli matrices are
interchangeably referred to as (0, 1, 2, 3) or (_, X, Y, Z).
"""

##### Imports #####
import numpy as np

###################

# These arrays are used to find products of Pauli matrices
RULES = np.array([1, 3, 1, 3])
SIGN_RULES = np.array([[1, 1, 1, 1],
                       [1, 1, 1j, -1j],
                       [1, -1j, 1, 1j],
                       [1, 1j, -1j, 1]])


def product(sigma1: int, sigma2: int) -> int:
    """
    Finds the unsigned product of two Pauli matrices.

    Parameters:
    -----------
    sigma1: int and sigma2: int
        The two Pauli matrices. The integer can be between 0 and 3.

    Returns:
    --------
    product: int
        The product of two Pauli matrices. The product of any matrix with itself gives 0. The product of any matrix with
        0 gives the same matrix. The product of any two different non-identity matrices returns the third non-identity
        matrix
    """

    return (sigma1 + sigma2 * RULES[sigma1]) % 4


def signed_product(sigma1: int, sigma2: int) -> tuple[int, complex]:
    """
    Finds the signed product of two Pauli matrices.

    Parameters:
    -----------
    sigma1: int
        The first Pauli matrix. The integer can be between 0 and 3.
    sigma2: int
        The second Pauli matrix. The integer can be between 0 and 3.

    Returns:
    --------
    product: int
        The unsigned product of two Pauli matrices.
    sign: complex
        The sign of the product. If two non-identity matrices are multiplied, the sign will be +i if :math:`\sigma_1
        \sigma_2 = c \sigma_3` is a cyclic permutation of (X, Y, Z) and -i if it is not.
    """

    return product(sigma1, sigma2), SIGN_RULES[sigma1, sigma2]


def string_product(string1: tuple[int], string2: tuple[int]) -> tuple[tuple[int], complex, bool]:
    """
    Returns the signed product of two Pauli strings and whether they commute.

    Parameters:
    -----------
    string1: tuple[int]
        The first Pauli string.
    string2: tuple[int]
        The second Pauli string.

    Returns:
    --------
        result: tuple[int]
            The product of the two strings.
        sign: complex
            The sign of the product.
        commutator: bool
           True if the two strings commute and False otherwise.
    """

    # Consistency check
    if len(string1) != len(string2):
        raise Exception(f"Dimension mismatch ({len(string1)} and {len(string2)})")

    string1_array = np.array(string1)
    string2_array = np.array(string2)
    sign = np.prod(SIGN_RULES[string1_array, string2_array])
    result = tuple((string1_array + np.multiply(string2_array, RULES[string1_array])) % 4)

    if sign.imag == 0:
        return result, sign, True
    else:
        return result, sign, False


def mut_irr(n: int, x: float = np.pi) -> list[float]:
    """Returns a list of n mutually irrational numbers. Takes the optional argument x irrational to build the set"""

    y = x % 1
    numbers = [y] * n

    for i in range(1, n):
        y = (x * y) % 1
        numbers[i] = y

    return numbers


def strings_to_dict(strings: list[tuple[int]] | tuple[int], coefficients: list[complex] | complex) -> dict[
    tuple[int], complex]:
    """
    Returns a dictionary for a Pauli word with the Pauli strings as keys and their corresponding coefficients as values.

    Parameters:
    -----------
    strings: list[tuple[int]] or tuple[int]:
        Can take one Pauli string or a list of Pauli strings.
    coefficients: list[complex] or complex:
        Can take one coefficient or a list of coefficients.

    Returns:
    --------
    dict[tuple[int], complex]:
        The Pauli word as a dictionary.
    """

    # Data reformatting
    coefficients_array = np.array([coefficients]).flatten()
    strings_array = np.array([strings])

    if len(strings_array.shape) > 2:
        strings_array = np.squeeze(strings_array, axis=0)

    # Consistency check
    if len(strings_array) != len(coefficients_array):
        raise Exception(f"Length mismatch - strings: {len(strings_array)}, coefficients: {len(coefficients_array)}")

    strings_array = tuple(map(tuple, strings_array))
    return dict(zip(strings_array, coefficients_array))


def full_sum(word_dict1: dict[tuple[int], complex], word_dict2: dict[tuple[int], complex], tol: float = 0) -> dict[
    tuple[int], complex]:
    """
    Finds the sum of two Pauli words.

    Parameters:
    -----------
    word_dict1: dict[tuple[int], complex]
        The first Pauli word.
    word_dict2: dict[tuple[int], complex]
        The second Pauli word.
    tol: float
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0. Default tolerance
        is 0.

    Returns:
    --------
    result: dict[tuple[int], complex]
        The sum of the two Pauli words as a dictionary.
    """

    result = word_dict1.copy()
    for key in word_dict2.keys():
        result[key] = result.get(key, 0) + word_dict2[key]
        if abs(result[key]) <= tol:
            result.pop(key)
    return result


def full_product(word_dict1: dict[tuple[int], complex], word_dict2: dict[tuple[int], complex], tol: float = 0) -> \
        dict[tuple[int], complex]:
    """
    Finds the product of two Pauli words.

    Parameters:
    -----------
    word_dict1: dict[tuple[int], complex]
        The first Pauli word.
    word_dict2: dict[tuple[int], complex]
        The second Pauli word.
    tol: float
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0. Default tolerance
        is 0.

    Returns:
    --------
    result: dict[tuple[int], complex]
        The product of the two Pauli words as a dictionary.
    """

    result = {}
    for key1 in word_dict1.keys():
        for key2 in word_dict2.keys():
            string, sign, c = string_product(key1, key2)
            result[string] = result.get(string, 0) + sign * word_dict1[key1] * word_dict2[key2]
            if abs(result[string]) <= tol:
                result.pop(string)

    return result


def string_exp(string: tuple[int], angle: float) -> dict[tuple[int], complex]:
    """
    Finds the exponential of a Pauli string :math:`\mathrm{e}^{\mathrm{i} x P} = \cos{x} + \mathrm{i}P\sin{x}`.

    Parameters:
    -----------
    string: tuple[int]
        The Pauli string to be exponentiated.
    angle: float
        The angle of rotation.

    Returns:
    --------
    result: dict[tuple[int], complex]
        The resulting Pauli word as a dictionary.
    """

    result = {}
    if np.cos(angle) != 0:
        result[(0,) * len(string)] = np.cos(angle)
    if np.sin(angle) != 0:
        result[string] = 1j * np.sin(angle)
    return result


def exp_conjugation(generators: list[tuple[int]] | tuple[int], angles: list[float] | float,
                    word_dict: dict[tuple[int], complex], tol: float = 0) -> dict[tuple[int], complex]:
    """
    Returns the conjugation of a Pauli word :math:`\mathrm{e}^{\mathrm{i} x_{1} P_1} ... \mathrm{e}^{\mathrm{i} x_n P_n}
    X \mathrm{e}^{-\mathrm{i} x_{n} P_n} ... \mathrm{e}^{-\mathrm{i} x_1 P_1}`.
    
    Parameters:
    -----------
    generators: list[tuple[int]] | tuple[int]
        Can take a one Pauli string or a list of Pauli strings to be exponentiated.
    angles: list[float] | float
        Can take one angle or a list of angles.
    word_dict: dict[tuple[int], complex]
        The Pauli word to be conjugated.
    tol: float
        Tolerance. Non-negative number. Any value less than or equal to the tolerance is considered 0. Default tolerance
        is 0.
    """

    result = {}
    # Data reformatting
    angles_array = np.array([angles]).flatten()
    cosine_array = np.cos(2 * angles_array)
    sine_array = np.sin(2 * angles_array)
    generators_array = np.array([generators])

    if len(generators_array.shape) > 2:
        generators_array = np.squeeze(generators_array, axis=0)

    # Consistency check
    if len(generators_array) != len(angles_array):
        raise Exception(f"Length mismatch - generators: {len(generators_array)}, angles: {len(angles_array)}")

    for key in word_dict.keys():
        coefficient = word_dict[key]
        for i in range(len(angles_array)):
            string, sign, c = string_product(tuple(generators_array[i]), key)

            # If the ith exponent commutes with string (key) in the Pauli word do nothing
            if c:
                result[key] = result.get(key, 0) + coefficient
            # If it doesn't commute it necessary anticommutes. Perform the operation exp(2ixP).string
            else:
                result[key] = result.get(key, 0) + cosine_array[i] * coefficient
                result[string] = result.get(string, 0) + sign * 1j * sine_array[i] * coefficient

                if abs(result[string]) <= tol:
                    result.pop(string)
        if abs(result[key]) <= tol:
            result.pop(key)

    return result


def trace(word_dict: dict[tuple[int], complex]) -> float | complex:
    """
    Finds the normalized trace of a Pauli word.
    
    Parameters:
    -----------
    word_dict: dict[tuple[int], complex]
        The Pauli word.

    Returns:
    --------
    trace: float | complex
        The trace of the Pauli word divided by the length of a Pauli string.
    """

    identity = (0,) * len(next(iter(word_dict)))
    return word_dict.get(identity, 0)
