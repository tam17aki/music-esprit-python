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
from scipy.signal import find_peaks

from .base import MusicAnalyzerBase
from mixins.covariance import ForwardBackwardMixin


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
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        n_samples = signal.size
        subspace_dim = n_samples // 3
        model_order = 2 * self.n_sinusoids
        if subspace_dim <= model_order:
            warnings.warn(
                "Invalid subspace dimension for MUSIC. Returning empty result."
            )
            return np.array([])

        # 1. Estimate the noise subspace
        noise_subspace = self._estimate_noise_subspace(
            signal, subspace_dim, model_order
        )
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace. Returning empty result.")
            return np.array([])

        freq_grid, music_spectrum = self._calculate_music_spectrum(
            subspace_dim, noise_subspace
        )
        # 3. Detecting peaks from a spectrum
        estimated_freqs = self._find_music_peaks(freq_grid, music_spectrum)
        return estimated_freqs

    def _calculate_music_spectrum(
        self,
        subspace_dim: int,
        noise_subspace: npt.NDArray[np.complex128],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the MUSIC pseudospectrum over a frequency grid."""
        freq_grid: npt.NDArray[np.float64] = np.linspace(
            0, self.fs / 2, num=self.n_grids, dtype=np.float64
        )
        omegas = 2 * np.pi * freq_grid / self.fs
        l_vector = np.arange(subspace_dim).reshape(-1, 1)
        steering_matrix = np.exp(-1j * l_vector @ omegas.reshape(1, -1))
        temp_matrix = noise_subspace.conj().T @ steering_matrix
        denominator_values = np.einsum("ij,ji->i", temp_matrix.conj().T, temp_matrix)
        music_spectrum = 1 / (np.abs(denominator_values) + 1e-12)
        return freq_grid, music_spectrum

    def _find_music_peaks(
        self,
        freq_grid: npt.NDArray[np.float64],
        music_spectrum: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Find the N strongest peaks from the MUSIC spectrum."""
        # 1. Find all "local maxima" as peak candidates.
        #    Ignores extremely small noise floor fluctuations.
        all_peaks, _ = find_peaks(
            music_spectrum,
            height=np.median(music_spectrum),
            prominence=np.std(music_spectrum) / 2.0,
        )
        all_peaks = np.array(all_peaks, dtype=np.int64)
        if all_peaks.size < self.n_sinusoids:
            return freq_grid[all_peaks] if all_peaks.size > 0 else np.array([])

        # 2. From all the peak candidates found, select N peaks
        #    with the highest spectral values.
        strongest_peak_indices = all_peaks[
            np.argsort(music_spectrum[all_peaks])[-self.n_sinusoids :]
        ]
        estimated_freqs = freq_grid[strongest_peak_indices]

        return np.sort(estimated_freqs)


@final
class SpectralMusicAnalyzerFB(ForwardBackwardMixin, SpectralMusicAnalyzer):
    """Spectral MUSIC analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """

    pass
