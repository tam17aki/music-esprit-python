# -*- coding: utf-8 -*-
"""The definition of RootMinNormAnalyzer class.

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

import warnings
from typing import final, override

import numpy as np
import numpy.typing as npt

from mixins.covariance import ForwardBackwardMixin

from .._common import find_freqs_from_roots
from .base import MinNormAnalyzerBase


class RootMinNormAnalyzer(MinNormAnalyzerBase):
    """Parameter analyzer using the Root Min-Norm algorithm."""

    def __init__(self, fs: float, n_sinusoids: int, sep_factor: float = 0.4):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
        """
        super().__init__(fs, n_sinusoids)
        self.sep_factor: float = sep_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids using Root-MinNorm.

        This method overrides the abstract method in the base class.

        Args:
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Calculate the minimum norm vector `d` from the noise subspace
        min_norm_vector = self._calculate_min_norm_vector(signal)
        if min_norm_vector is None:
            warnings.warn(
                "Failed to compute the Min-Norm vector. Returning empty result."
            )
            return np.array([])

        # 2. Estimate frequencies by finding the roots of a polynomial with
        #    coefficients `d`
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        estimated_freqs = find_freqs_from_roots(
            self.fs, self.n_sinusoids, min_norm_vector, min_separation_hz
        )
        return estimated_freqs


@final
class RootMinNormAnalyzerFB(ForwardBackwardMixin, RootMinNormAnalyzer):
    """Root Min-Norm analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
