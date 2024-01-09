from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class SignalAnalysis:
    """
    A class for analyzing response function data. It can be used to perform a Fourier transform of the data, find peaks,
    calculate the density of states, find the projected density of states, and self-energy.
    """

    def __init__(
        self, signal: NDArray, x_axis: NDArray, domain: Literal["time", "frequency"]
    ):
        self.signal = signal
        self.domain = domain
        match domain:
            case "time":
                self.times = x_axis

            case "frequency":
                self.frequencies = x_axis

            case _:
                raise ValueError(
                    f'Expected `domain` to be "time" or "frequency", but found {domain}'
                )

    def fourier_transform(self) -> tuple[NDArray, NDArray]:
        ...

    def locate_peaks(
        self, min_height: float = 0.1, min_threshold: float = 1e2
    ) -> NDArray: # type: ignore
        """
        Locates peaks in the signal and returns the frequency values corresponding to the peak locations.

        Parameters:
        -----------
        min_height: float
            The minimum height at which a peak can be located.
        min_threshold: float
            The minimum vertical distance to the neighboring points at which a peak can be located.

        Returns:
        --------
        peaks: NDArray
            The indices of the peaks within the signal.
        """
        if self.domain == "time":
            print(
                "Variable `signal` was initialized in the time domain. Calculating Fourier transform to locate peaks..."
            )
            self.fft_signal, self.frequencies = self.fourier_transform()
            self.peak_locations, _ = find_peaks(
                self.fft_signal, height=min_height, threshold=min_threshold
            )

            peak_frequency_values = np.array(
                [self.frequencies[location] for location in self.peak_locations]
            )
            return peak_frequency_values

        elif self.domain == "frequency":
            self.peak_locations, _ = find_peaks(
                self.signal, height=min_height, threshold=min_threshold
            )
            peak_frequency_values = np.array(
                [self.frequencies[location] for location in self.peak_locations]
            )
            return peak_frequency_values

    def get_density_of_states(self):
        ...

    def get_projected_density_of_states(self):
        ...

    def get_self_energy(self):
        ...
