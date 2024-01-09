from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class GreensFunction:
    times: NDArray

    frequencies: NDArray

    def __init__(
        self,
        main_axis: NDArray,
        kind: Literal["time", "frequency"],
        N_sample_fft: int = 1000,
        dt_fft: float = 0.01,
        representation: Literal["lehmann", "operator"] = "lehmann",
    ):
        match kind:
            case "time":
                self.times = main_axis
                self.frequencies = fftfreq(N_sample_fft, dt_fft)

            case "frequency":
                self.frequencies = main_axis
                self.times = np.linspace(0, N_sample_fft * dt_fft, N_sample_fft)

            case _:
                raise ValueError(
                    'Expected `kind` to be "time" or ' f'"frequency", but found {kind}'
                )

        match representation:
            case "lehmann":
                self.greensfunction = self.lehmann_representation()

            case "operator":
                self.greensfunction = self.operator_representation()

            case _:
                raise ValueError(
                    f'Expected `representation` to be "lehmann" or "operator", but found {representation}'
                )

    def lehmann_representation(self):
        ...

        # TODO: calculate Lehmann representation

    def operator_representation(self):
        ...

        # TODO: calculate operator representation

    def spectral_representation(self):
        ...

        # TODO: calculate spectral representation

    def peak_locator(self):
        ...
        # peaks = find_peaks(self.times)
