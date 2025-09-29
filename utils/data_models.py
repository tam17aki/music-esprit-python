# -*- coding: utf-8 -*-
"""Defines core data structures for the project using dataclasses.

This module centralizes the definitions of immutable data models that
are shared across different parts of the application, such as the signal
generator and the analyzers.

Copyright (C) 2025 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SinusoidParameters:
    """A class to store the parameters of multiple sinusoids."""

    frequencies: npt.NDArray[np.float64]
    amplitudes: npt.NDArray[np.float64]
    phases: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentConfig:
    """A class to store the configuration for an experiment."""

    fs: float
    duration: float
    snr_db: float
    freqs_true: npt.NDArray[np.float64]
    amp_range: tuple[float, float]

    @property
    def n_sinusoids(self) -> int:
        """Return the number of sinusoids."""
        return self.freqs_true.size
