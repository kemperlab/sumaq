from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class SignalAnalysis:

    def __init__(self):
        ...

    def fourier_transform(self):
        ...

    def peak_locator(self):
        ...

    def get_density_of_states(self):
        ...

    def get_projected_density_of_states(self):
        ...

    def get_self_energy(self):
        ...
