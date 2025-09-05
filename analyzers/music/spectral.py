# -*- coding: utf-8 -*-
"""The definition of SpectralMusicAnalyzer class.

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
from .base import MusicAnalyzerBase


class SpectralMusicAnalyzer(MusicAnalyzerBase):
    """MUSIC analyzer using spectral peak picking."""

    def __init__(self, fs: float, n_sinusoids: int, n_grids: int):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            n_grids (int, optional): Number of grid points for MUSIC algorithm.
        """
        super().__init__(fs, n_sinusoids)
        self.n_grids: int = n_grids

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Estimate the noise subspace
        noise_subspace = self._estimate_noise_subspace(signal)
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace. Returning empty result.")
            return np.array([])

        # 2. Calculate MUSIC pseudospectrum
        freq_grid, music_spectrum = self._calculate_music_spectrum(noise_subspace)

        # 3. Detecting peaks from a spectrum
        estimated_freqs = find_peaks_from_spectrum(
            self.n_sinusoids, freq_grid, music_spectrum
        )
        return estimated_freqs

    def _calculate_music_spectrum(
        self, noise_subspace: npt.NDArray[np.complex128]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the MUSIC pseudospectrum over a frequency grid.

        Args:
            noise_subspace (np.ndarray): The noise subspace matrix (complex128).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - freq_grid (np.ndarray): Frequency grid (float64).
                - music_spectrum (np.ndarray): MUSIC pseudospectrum (float64).
        """
        freq_grid: npt.NDArray[np.float64] = np.linspace(
            0, self.fs / 2, num=self.n_grids, dtype=np.float64
        )
        omegas = 2 * np.pi * freq_grid / self.fs
        l_vector = np.arange(self.subspace_dim).reshape(-1, 1)
        steering_matrix = np.exp(-1j * l_vector @ omegas.reshape(1, -1))
        temp_matrix = noise_subspace.conj().T @ steering_matrix
        denominator_values = np.einsum("ij,ji->i", temp_matrix.conj().T, temp_matrix)
        music_spectrum = 1 / (np.abs(denominator_values) + 1e-12)
        return freq_grid, music_spectrum


@final
class SpectralMusicAnalyzerFB(ForwardBackwardMixin, SpectralMusicAnalyzer):
    """Spectral MUSIC analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
