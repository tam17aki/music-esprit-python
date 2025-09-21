# -*- coding: utf-8 -*-
"""Defines FastMusicAnalyzer class for FAST MUSIC algorithm.

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

from .._common import find_peaks_from_spectrum
from .base import MusicAnalyzerBase


@final
class FastMusicAnalyzer(MusicAnalyzerBase):
    """Analyzes sinusoidal parameters using the FAST MUSIC algorithm.

    This analyzer is specialized for periodic or approximately periodic signals.
    It leverages the property that the autocorrelation matrix of a periodic
    signal can be approximated by a circulant matrix, which allows replacing
    the computationally expensive eigenvalue decomposition (EVD) with a
    Fast Fourier Transform (FFT).

    This approach can be significantly faster than standard MUSIC methods
    for signals with a detectable periodicity, such as musical tones or
    voiced speech.

    Reference:
        O. Das, J. S. Abel, J. O. Smith III, "FAST MUSIC – An Efficient
        Implementation of The Music Algorithm For Frequency Estimation of
        Approximately Periodic Signals," International Conference on Digital
        Audio Effects (DAFx), vol.21, pp.342-348, 2018.
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        n_grids: int = 16384,
        min_freq_period: float = 20.0,
    ):
        """Initialize the FAST MUSIC analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            n_grids (int, optional):
                The number of points for the final pseudospectrum search grid.
                Defaults to 16384.
            min_freq_period (float, optional):
                The minimum frequency in Hz to consider when searching for the
                signal's fundamental period. This helps constrain the search
                range of the periodicity detection. Defaults to 20.0.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio=0.5)  # pass dummy
        self.n_grids = n_grids
        self.min_freq_period = min_freq_period

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        # 0. Convert the signal to real (FAST MUSIC often assumes real signals)
        if np.iscomplexobj(signal):
            warnings.warn(
                "FastMusicAnalyzer received a complex-valued signal. "
                + "This algorithm is designed for real-valued signals and will "
                + "discard the imaginary part. For complex signal analysis, "
                + "consider using other analyzers like StandardEspritAnalyzer."
            )
        real_signal = np.real(signal)

        # 1. Calculate the autocorrelation
        acf = np.correlate(real_signal, real_signal, mode="full")
        acf = acf[real_signal.size - 1 :].astype(np.float64)

        # 2. Detect the period M
        min_period = int(self.fs / self.min_freq_period)
        max_period = real_signal.size // 2
        period_m = self._find_period_amdf(acf, min_period, max_period)

        # 3. Identify the signal space indices from the power spectrum
        power_spectrum = np.abs(np.fft.fft(acf[:period_m]))
        num_unique_bins = period_m // 2 + 1
        half_spectrum = power_spectrum[:num_unique_bins]
        if half_spectrum.size < self.n_sinusoids:
            # Fallback if spectrum is too short
            signal_space_indices = np.argsort(half_spectrum)[::-1]
        else:
            signal_space_indices = np.argsort(half_spectrum)[::-1][: self.n_sinusoids]

        # 4. Calculates the FAST MUSIC pseudospectrum in closed form
        freq_grid, pseudospectrum = self._calculate_fast_music_spectrum(
            period_m, signal_space_indices
        )

        # 5. Find peaks and estimate frequencies (reuse common helpers)
        return find_peaks_from_spectrum(pseudospectrum, self.n_sinusoids, freq_grid)

    @staticmethod
    def _find_period_amdf(
        acf: npt.NDArray[np.float64], min_period: int, max_period: int
    ) -> int:
        """Estimates the fundamental period of a signal from its ACF.

        This method uses the Average Magnitude Difference Function (AMDF)
        to find the most prominent period within a specified range.

        Args:
            acf (np.ndarray):
                The autocorrelation function of the signal (float64).
            min_period (int):
                The minimum period (in samples) to search for.
            max_period (int):
                The maximum period (in samples) to search for.

        Returns:
            int: The estimated period in samples.
        """
        lags = range(min_period, max_period)
        amdf = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            diff = np.abs(acf[lag:] - acf[:-lag])
            amdf[i] = np.mean(diff)

        normalized_amdf = amdf / (np.array(lags) + 1e-9)
        inv_amdf = -normalized_amdf
        peaks, _ = find_peaks(inv_amdf)

        if len(peaks) > 0:
            best_peak_index = peaks[np.argmax(inv_amdf[peaks])]
            return int(np.array(lags)[best_peak_index])
        return (min_period + max_period) // 2

    def _calculate_fast_music_spectrum(
        self, period_m: int, signal_space_indices: npt.NDArray[np.int_]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the FAST MUSIC pseudospectrum using a closed-form expression.

        Instead of a direct spectral search, this method computes the
        pseudospectrum using a sum of squared aliased sinc functions, based on
        the signal eigenvector frequencies identified from the ACF's power spectrum.

        Args:
            period_m (int):
                The estimated fundamental period of the signal (M).
            signal_space_indices (np.ndarray):
                Indices of the power spectrum peaks corresponding to the
                signal subspace eigenvectors (int_).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing the frequency grid (in Hz) and the
                calculated FAST MUSIC pseudospectrum (float64 and float64).
        """
        k = np.arange(self.n_grids).reshape(1, -1)  # (1, N_search)
        mi = signal_space_indices.reshape(-1, 1)  # (M, 1)
        x = k / self.n_grids - mi / period_m  # (M, N_search)
        asinc_matrix = self._asinc(x, period_m) ** 2
        asinc_sum = np.sum(asinc_matrix, axis=0)
        denominator = period_m - (1 / period_m) * asinc_sum
        pseudospectrum = 1 / (np.abs(denominator) + 1e-12)
        freq_grid = np.arange(self.n_grids) * self.fs / self.n_grids
        return freq_grid, pseudospectrum

    @staticmethod
    def _asinc(x: npt.NDArray[np.float64], m: int) -> npt.NDArray[np.float64]:
        """Calculate the aliased sinc function (Dirichlet kernel).

        This function is defined as sin(pi*m*x) / sin(pi*x). It handles the
        singularity at x = 0, where the value is m.

        Args:
            x (np.ndarray): Input array (float64).
            m (int): The length of the sequence (period).

        Returns:
            np.ndarray: The result of the aliased sinc function (float64).
        """
        epsilon = 1e-12
        numerator = np.sin(np.pi * m * x)
        denominator = np.sin(np.pi * x)
        near_zero_den = np.abs(denominator) < epsilon
        result: npt.NDArray[np.float64] = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=~near_zero_den
        )
        result[near_zero_den] = m
        return result
