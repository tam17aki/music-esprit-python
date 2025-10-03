# -*- coding: utf-8 -*-
"""Defines CfhAnalyzer class for the CFH algorithm.

This module implements the Coarse-to-fine Hybrid Aboutanios and Mulgrew
and q-shift (CFH) estimator for sinusoidal parameter analysis.

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

from typing import final, override

import numpy as np

from utils.data_models import ComplexArray, InterpolatorType

from .._common import ZERO_LEVEL
from .base import IterativeAnalyzerBase

# Minimum number of samples required for 3-point interpolation.
_MIN_SAMPLES_FOR_INTERPOLATION = 3


@final
class CfhAnalyzer(IterativeAnalyzerBase):
    """Analyzes sinusoids using an iterative interpolation method.

    This analyzer uses the iterative signal subtraction methodology from
    the RELAX algorithm. It employs Candan's 3-point DFT interpolation
    for precise frequency estimation at each step, avoiding the need for
    large zero-padded FFTs.
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        interpolator: InterpolatorType = "haqse",
    ):
        """Initialize the analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids to estimate.
            interpolator (InterpolatorType, optional):
                The DFT interpolation method to use. Can be "candan" or
                "haqse". Defaults to "haqse".
        """
        super().__init__(fs, n_sinusoids)
        self.interpolator = interpolator
        self._single_freq_estimator = (
            self._estimate_single_freq_haqse
            if interpolator == "haqse"
            else self._estimate_single_freq_candan
        )

    @override
    def _estimate_single_frequency(self, signal: ComplexArray) -> float | None:
        """Delegate to the selected interpolator method."""
        return self._single_freq_estimator(signal)

    def _estimate_single_freq_haqse(
        self, signal: ComplexArray
    ) -> float | None:
        """Estimates a single frequency via a HAQSE-family interpolator.

        This method implements a non-iterative, high-accuracy frequency
        interpolator based on the principles of HAQSE and Serbes. It
        uses three DFT samples around the coarse peak to calculate a
        refined frequency offset.

        Args:
            signal: The complex-valued input signal or residual.

        Returns:
            The estimated frequency in Hz, or None if estimation fails.
        """
        n = signal.size
        if n < _MIN_SAMPLES_FOR_INTERPOLATION:
            return None

        # 1. Coarse search: Find the integer index of the DFT peak
        dft_sig = np.fft.fft(signal)

        # Search in positive frequencies, excluding DC and Nyquist
        # boundaries
        search_space = np.abs(dft_sig[1 : n // 2])
        if search_space.size == 0:
            return None
        k_c = int(np.argmax(search_space)) + 1  # Coarse peak index

        # Interpolation is not possible at the spectrum boundaries
        if k_c <= 0 or k_c >= (n // 2) - 1:
            return (k_c / n) * self.fs

        # 2. Fine estimation using 3 complex DFT samples
        x_km1 = dft_sig[k_c - 1]
        x_k = dft_sig[k_c]
        x_kp1 = dft_sig[k_c + 1]

        # --- HAQSE/Serbes-style correction term calculation ---
        # δ = Re{ (X_{k-1} - X_{k+1}) / (2*X_k + X_{k-1} + X_{k+1}) }

        # Calculate numerator and denominator using complex arithmetic
        # to reduce local variables.
        term_a = x_km1 - x_kp1
        term_b = 2 * x_k + x_km1 + x_kp1

        if abs(term_b) < ZERO_LEVEL:
            delta = 0.0
        else:
            # Re{ a / b } = (Re{a}Re{b} + Im{a}Im{b}) / |b|²
            # This is equivalent to Re{ a * conj(b) } / |b|²
            # np.real(term_a * np.conj(term_b)) is more direct
            numerator = np.real(term_a * np.conj(term_b))
            denominator = np.real(term_b * np.conj(term_b))  # |b|^2
            delta = numerator / denominator

        # The refined frequency is k_c + delta (in bins)
        # Clamp the correction to be within [-0.5, 0.5] for stability
        delta = np.clip(delta, -0.5, 0.5)

        refined_freq_norm = (k_c + delta) / n

        return float(refined_freq_norm * self.fs)

    def _estimate_single_freq_candan(
        self, signal: ComplexArray
    ) -> float | None:
        """Estimate a single frequency via Candan's interpolation.

        This method uses a robust 3-point DFT sample interpolation
        technique to achieve sub-bin frequency accuracy.

        Args:
            signal: The complex-valued input signal or residual.

        Returns:
            The estimated frequency in Hz, or None if estimation fails.
        """
        n = signal.size
        if n < _MIN_SAMPLES_FOR_INTERPOLATION:
            return None

        # 1. Coarse search: Find the DFT peak and its neighbors
        dft_sig = np.fft.fft(signal)
        # Search in positive frequencies, excluding DC and Nyquist.
        search_space = np.abs(dft_sig[1 : n // 2])
        if search_space.size == 0:
            return None
        k_c = int(np.argmax(search_space)) + 1

        # Interpolation is not possible at the spectrum boundaries.
        if k_c <= 0 or k_c >= (n // 2) - 1:
            return (k_c / n) * self.fs

        # 2. Fine estimation using 3 complex DFT samples
        x_kc = dft_sig[k_c]
        x_km1 = dft_sig[k_c - 1]
        x_kp1 = dft_sig[k_c + 1]
        if abs(x_kc) < ZERO_LEVEL:
            return (k_c / n) * self.fs

        alpha = x_km1 / x_kc
        beta = x_kp1 / x_kc

        denominator = 2.0 - alpha - beta
        if np.abs(denominator) < ZERO_LEVEL:
            delta = 0.0
        else:
            delta = np.real((alpha - beta) / denominator)

        # 3. Calculate final frequency from the interpolated index
        refined_freq_norm = (k_c + delta) / n

        return refined_freq_norm * self.fs
