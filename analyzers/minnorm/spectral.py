# -*- coding: utf-8 -*-
"""Defines SpectralMinNormAnalyzer class for Spectral Min-Norm.

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
from numpy.fft import fft, fftfreq

from mixins.covariance import ForwardBackwardMixin

from .._common import find_peaks_from_spectrum
from ..models import AnalyzerParameters
from .base import MinNormAnalyzerBase


class SpectralMinNormAnalyzer(MinNormAnalyzerBase):
    """Parameter analyzer using the Spectral Min-Norm algorithm."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        n_grids: int = 16384,
        subspace_ratio: float = 1 / 3,
    ) -> None:
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            n_grids (int, optional):
                Number of grid points for Spectral Min-Norm.
            subspace_ratio (float, optional):
                The ratio of the subspace dimension to the signal
                length. Must be between 0 and 0.5. Defaults to 1/3.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio)
        self.n_grids: int = n_grids

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multi-sinusoids via Spec. Min-Norm.

        This method overrides the abstract method in the base class.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Estimate the noise subspace (reusing the base class method)
        noise_subspace = self._estimate_noise_subspace(signal)
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace.")
            return np.array([])

        # 2. Calculate the minimum norm vector `d` from the noise
        #    subspace
        min_norm_vector = self._calculate_min_norm_vector(noise_subspace)
        if min_norm_vector is None:
            warnings.warn("Failed to compute the Min-Norm vector.")
            return np.array([])

        # 3. Calculate Min-Norm spectrum using `d`
        freq_grid, min_norm_spectrum = self._calculate_min_norm_spectrum(
            min_norm_vector
        )

        # 4. Searching for peaks in the spectrum
        estimated_freqs = find_peaks_from_spectrum(
            min_norm_spectrum, self.n_sinusoids, freq_grid
        )

        return estimated_freqs

    def _calculate_min_norm_spectrum(
        self,
        min_norm_vector: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the Min-Norm pseudospectrum over a frequency grid.

        Args:
            min_norm_vector (np.ndarray):
                The minimum norm vector (float64 or complex128).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - freq_grid: Frequency grid (float64).
                - min_norm_spectrum: Min-Norm pseudospectrum (float64).
        """
        # 1. Calculate the FFT of mininum norm vector
        fft_noise_eigvec = fft(min_norm_vector, n=self.n_grids)

        # 2. Calculate the power spectrum of FFT result
        power_spectra_noise = np.abs(fft_noise_eigvec) ** 2

        # 3. Calculate the MUSIC pseudospectrum
        music_spectrum = 1 / (power_spectra_noise + 1e-12)

        # 4. Build a frequency grid
        freq_grid = fftfreq(self.n_grids, d=1 / self.fs).astype(np.float64)

        # 5. Make a mask that only handles positive frequencies
        positive_freq_mask = freq_grid >= 0
        freq_grid = freq_grid[positive_freq_mask]
        music_spectrum = music_spectrum[positive_freq_mask]

        return freq_grid, music_spectrum

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include the 'n_grids'
        parameter specific to the Spectral Min-Norm method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and spectral-specific
                hyperparameters.
        """
        params = super().get_params()
        params["n_grids"] = self.n_grids
        return params


@final
class SpectralMinNormAnalyzerFB(ForwardBackwardMixin, SpectralMinNormAnalyzer):
    """Spectral Min-Norm analyzer enhanced with FB averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix
    calculation.
    """
