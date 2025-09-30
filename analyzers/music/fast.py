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

from .._common import (
    ZERO_LEVEL,
    find_peak_indices_from_spectrum,
    find_peaks_from_spectrum,
)
from ..models import AnalyzerParameters
from .base import MusicAnalyzerBase


@final
class FastMusicAnalyzer(MusicAnalyzerBase):
    """Implements the FAST MUSIC algorithm for frequency estimation.

    This analyzer is specialized for periodic or approximately periodic
    signals.  It leverages the property that the autocorrelation matrix
    of a periodic signal can be approximated by a circulant matrix,
    allowing the computationally expensive eigenvalue decomposition
    (EVD) to be replaced by an FFT.

    This approach can be significantly faster than standard MUSIC
    methods for signals with a detectable periodicity, such as musical
    tones or voiced speech.

    Reference:
        O. Das, J. S. Abel, J. O. Smith III, "FAST MUSIC - An Efficient
        Implementation of The Music Algorithm For Frequency Estimation
        of Approximately Periodic Signals," International Conference on
        Digital Audio Effects (DAFx), vol.21, pp.342-348, 2018.
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
                The number of points for the final pseudospectrum search
                grid.  Defaults to 16384.
            min_freq_period (float, optional):
                The minimum frequency in Hz to consider when searching
                for the signal's fundamental period. This helps
                constrain the search range of the periodicity
                detection. Defaults to 20.0.
        """
        super().__init__(fs, n_sinusoids)
        self.n_grids = n_grids
        self.min_freq_period = min_freq_period

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids using FAST MUSIC.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 0. Convert the signal to real (FAST MUSIC often assumes real
        #    signals)
        if np.iscomplexobj(signal):
            warnings.warn(
                "FastMusicAnalyzer received a complex-valued signal. "
                + "This algorithm is designed for real-valued signals and will"
                + " discard the imaginary part. For complex signal analysis, "
                + "consider using other analyzers like StandardEspritAnalyzer."
            )
        real_signal = np.real(signal)

        # 1. Calculate the autocorrelation
        acf = np.correlate(real_signal, real_signal, mode="full").astype(
            np.float64
        )
        acf = acf[real_signal.size - 1 :]

        # 2. Detect the period M
        min_period = int(self.fs / self.min_freq_period)
        max_period = real_signal.size // 2
        period_m = self._find_period(acf, min_period, max_period)

        # 3. Calculate the amplitude spectrum
        amplitude_spectrum = np.abs(np.fft.fft(acf[:period_m]))
        num_unique_bins = period_m // 2 + 1
        half_spectrum = amplitude_spectrum[:num_unique_bins]

        # 4. Search peaks on the spectrum and identify the signal space
        #    indices
        signal_space_indices = find_peak_indices_from_spectrum(
            half_spectrum, self.n_sinusoids
        )

        # 5. Calculates the FAST MUSIC pseudospectrum in closed form
        freq_grid, pseudospectrum = self._calculate_fast_music_spectrum(
            period_m, signal_space_indices
        )

        # 6. Find peaks and estimate frequencies (reuse common helpers)
        return find_peaks_from_spectrum(
            pseudospectrum, self.n_sinusoids, freq_grid
        )

    @staticmethod
    def _find_period(
        acf: npt.NDArray[np.float64], min_period: int, max_period: int
    ) -> int:
        """Estimate the fundamental period of a signal from its ACF.

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
        if min_period >= max_period:
            warnings.warn(
                "Invalid search range for period detection: "
                + f"min_period ({min_period}) must be less than "
                + f"max_period ({max_period}). Falling back to midpoint."
            )
            return (min_period + max_period) // 2

        search_range = acf[min_period:max_period]
        all_peaks, properties = find_peaks(
            search_range,
            height=np.max(search_range) * 0.1,
            prominence=np.std(search_range) * 0.25,
        )
        if all_peaks.size == 0:
            warnings.warn(
                "No periodic peaks found in ACF. "
                + "Using the highest point as fallback."
            )
            best_peak_local_index = np.argmax(search_range)
        elif all_peaks.size > 0 and "prominences" in properties:
            best_peak_local_index = all_peaks[
                np.argmax(properties["prominences"])
            ]
        else:
            best_peak_local_index = np.argmax(search_range)

        return int(min_period + best_peak_local_index)

    def _calculate_fast_music_spectrum(
        self, period_m: int, signal_space_indices: npt.NDArray[np.int_]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates the FAST MUSIC pseudospectrum.

        This method implements the core of the FAST MUSIC
        algorithm. Instead of performing a computationally expensive
        spectral search, it calculates the pseudospectrum directly using
        a closed-form formula derived from the properties of circulant
        matrices.

        The formula is based on a sum of squared aliased sinc functions,
        which are evaluated using the signal space indices (approximated
        eigenvector frequencies) identified from the power spectrum of
        the ACF.

        Args:
            period_m (int):
                The estimated fundamental period of the signal (M).
            signal_space_indices (np.ndarray):
                Indices of the power spectrum peaks corresponding to the
                signal subspace eigenvectors (int_).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing the frequency grid in Hz and the FAST
                MUSIC pseudospectrum (float64 and float64).
        """
        k = np.arange(self.n_grids).reshape(1, -1)  # (1, N_search)
        mi = signal_space_indices.reshape(-1, 1)  # (M, 1)
        x = k / self.n_grids - mi / period_m  # (M, N_search)
        asinc_matrix = self._asinc(x, period_m) ** 2
        asinc_sum = np.sum(asinc_matrix, axis=0)
        denominator = period_m - (1 / period_m) * asinc_sum
        pseudospectrum = 1 / (np.abs(denominator) + ZERO_LEVEL)
        freq_grid = np.arange(self.n_grids) * self.fs / self.n_grids
        return freq_grid, pseudospectrum

    @staticmethod
    def _asinc(x: npt.NDArray[np.float64], m: int) -> npt.NDArray[np.float64]:
        """Calculate the aliased sinc function (Dirichlet kernel).

        Defined as sin(pi*m*x) / sin(pi*x), this function handles the
        singularity at x = 0, where the value is m.

        Args:
            x (np.ndarray):
                Input array (float64).
            m (int):
                The length of the sequence (period).

        Returns:
            np.ndarray:
                The result of the aliased sinc function (float64).
        """
        numerator = np.sin(np.pi * m * x)
        denominator = np.sin(np.pi * x)
        near_zero_den = np.abs(denominator) < ZERO_LEVEL
        result: npt.NDArray[np.float64] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=~near_zero_den,
        )
        result[near_zero_den] = m
        return result

    @override
    def get_params(self) -> AnalyzerParameters:
        """Returns the analyzer's hyperparameters.

        Extends the base implementation to include the 'n_grids'
        parameter specific to the FAST MUSIC method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["n_grids"] = self.n_grids
        params["min_freq_period"] = self.min_freq_period
        return params
