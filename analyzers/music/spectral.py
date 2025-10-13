# -*- coding: utf-8 -*-
"""Defines SpectralMusicAnalyzer class for the Spectral MUSIC method.

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
from scipy.fft import fft, fftfreq

from mixins.covariance import ForwardBackwardMixin
from utils.data_models import ComplexArray, FloatArray, NumpyFloat, SignalArray

from .._common import ZERO_LEVEL, find_peaks_from_spectrum
from ..models import AnalyzerParameters
from .base import MusicAnalyzerBase


class SpectralMusicAnalyzer(MusicAnalyzerBase):
    """MUSIC analyzer using spectral peak picking."""

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
                Number of grid points for MUSIC algorithm.
                Defaults to 16384.
            subspace_ratio (float, optional):
                The ratio of the subspace dimension to the signal
                length. Must be between 0 and 0.5. Defaults to 1/3.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio)
        self.n_grids: int = n_grids

    @override
    def _estimate_frequencies(self, signal: SignalArray) -> FloatArray:
        """Estimate frequencies of multi-sinusoids using Spectral MUSIC.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            FloatArray: Estimated frequencies in Hz.
                Returns empty arrays on failure.
        """
        # 1. Estimate the noise subspace
        noise_subspace = self._estimate_noise_subspace(signal)
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace.")
            return np.array([])

        # 2. Calculate MUSIC pseudospectrum
        freq_grid, music_spectrum = self._calculate_music_spectrum(
            noise_subspace
        )

        # 3. Detecting peaks from a spectrum
        estimated_freqs = find_peaks_from_spectrum(
            music_spectrum, self.n_sinusoids, freq_grid
        )

        return estimated_freqs

    def _calculate_music_spectrum(
        self, noise_subspace: FloatArray | ComplexArray
    ) -> tuple[FloatArray, FloatArray]:
        """Calculate the MUSIC pseudospectrum over a frequency grid.

        Args:
            noise_subspace (FloatArray | ComplexArray):
                The noise subspace matrix.

        Returns:
            tuple[FloatArray, FloatArray]:
                - freq_grid: Frequency grid.
                - music_spectrum: MUSIC pseudospectrum.
        """
        # 1. Calculate the FFT of each noise eigenvector
        fft_noise_eigvec = fft(noise_subspace, n=self.n_grids, axis=0)

        # 2. Calculate the power spectrum of each FFT result
        power_spectra_noise = np.abs(fft_noise_eigvec) ** 2

        # 3. Sum the power for each frequency
        #    to get the denominator of the MUSIC psuedospectrum
        denominator_values = np.sum(power_spectra_noise, axis=1)

        # 4. Calculate the MUSIC pseudospectrum
        music_spectrum = 1 / (denominator_values + ZERO_LEVEL)

        # 5. Build a frequency grid
        freq_grid = fftfreq(self.n_grids, d=1 / self.fs).astype(NumpyFloat)

        # 6. Make a mask that only handles positive frequencies
        positive_freq_mask = freq_grid >= 0
        freq_grid = freq_grid[positive_freq_mask]
        music_spectrum = music_spectrum[positive_freq_mask]

        return freq_grid, music_spectrum

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include the 'n_grids'
        parameter specific to the Spectral MUSIC method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params["n_grids"] = self.n_grids
        return params


@final
class SpectralMusicAnalyzerFB(ForwardBackwardMixin, SpectralMusicAnalyzer):
    """Spectral MUSIC analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix
    calculation.
    """
