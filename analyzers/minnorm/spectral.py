# -*- coding: utf-8 -*-
"""The definition of SpectralMinNormAnalyzer class.

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

from .._common import find_peaks_from_spectrum
from .base import MinNormAnalyzerBase


class SpectralMinNormAnalyzer(MinNormAnalyzerBase):
    """Parameter analyzer using the Spectral Min-Norm algorithm."""

    def __init__(self, fs: float, n_sinusoids: int, n_grids: int):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            n_grids (int, optional): Number of grid points for Spectral-MinNorm.
        """
        super().__init__(fs, n_sinusoids)
        self.n_grids: int = n_grids

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids using Spectral-MinNorm.

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

        # 2. Calculate Min-Norm spectrum using `d`
        freq_grid, min_norm_spectrum = self._calculate_min_norm_spectrum(
            min_norm_vector
        )

        # 3. Searching for peaks in the spectrum
        # estimated_freqs = self._find_peaks(freq_grid, min_norm_spectrum)
        estimated_freqs = find_peaks_from_spectrum(
            self.n_sinusoids, freq_grid, min_norm_spectrum
        )

        return estimated_freqs

    def _calculate_min_norm_spectrum(
        self, min_norm_vector: npt.NDArray[np.complex128]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the Min-Norm pseudospectrum over a frequency grid."""
        freq_grid: npt.NDArray[np.float64] = np.linspace(
            0, self.fs / 2, num=self.n_grids, dtype=np.float64
        )
        omegas = 2 * np.pi * freq_grid / self.fs
        l_vector = np.arange(self.subspace_dim).reshape(-1, 1)
        steering_matrix = np.exp(-1j * l_vector @ omegas.reshape(1, -1))
        denominator_values = np.abs(min_norm_vector.conj().T @ steering_matrix)
        min_norm_spectrum = 1 / (denominator_values + 1e-12)
        return freq_grid, min_norm_spectrum


@final
class SpectralMinNormAnalyzerFB(ForwardBackwardMixin, SpectralMinNormAnalyzer):
    """Spectral Min-Norm analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
