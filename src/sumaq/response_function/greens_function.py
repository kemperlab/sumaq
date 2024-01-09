from typing import Literal

from numpy.typing import NDArray
# from numpy.fft import fft, fftfreq


class GreensFunction():
    times: NDArray
    time_values: NDArray

    frequencies: NDArray
    frequency_values: NDArray

    def __init__(self, main_axis: NDArray, main_axis_values: NDArray,
                 kind: Literal["time", "frequency"]):
        match kind:
            case "time":
                self.times = main_axis
                self.time_values = main_axis_values

                # TODO: self.frequencies =
                # TODO: self.frequency_values =

            case "frequency":
                self.frequencies = main_axis
                self.frequency_values = main_axis_values

                # TODO: self.times =
                # TODO: self.time_values =

            case _:
                raise ValueError('Expected `kind` to be "time" or '
                                 f'"frequency", but found {kind}')
