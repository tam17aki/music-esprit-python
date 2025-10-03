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
from ..models import AnalyzerParameters
from .base import IterativeAnalyzerBase

_MIN_SAMPLES_FOR_INTERPOLATION = 3
_TAYLOR_APPROXIMATION_THRESHOLD = 1e-4


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
        self.interpolator: InterpolatorType = interpolator
        if interpolator == "haqse":
            self._single_freq_estimator = self._estimate_single_freq_haqse
        elif interpolator == "candan":
            self._single_freq_estimator = self._estimate_single_freq_candan

    @override
    def _estimate_single_frequency(self, signal: ComplexArray) -> float | None:
        """Delegate to the selected interpolator method."""
        return self._single_freq_estimator(signal)

    def _estimate_single_freq_haqse(
        self, signal: ComplexArray
    ) -> float | None:
        """Estimate a frequency via the Hybrid A&M/QSE interpolator.

        This method implements the two-stage HAQSE interpolator based
        on the MATLAB implementation by A. Serbes. It first obtains a
        coarse offset using the Aboutanios-Mulgrew (A&M) interpolator,
        then refines it using a Q-Shift Estimator (QSE).

        Args:
            signal: The complex-valued input signal or residual.

        Returns:
            The estimated frequency in Hz, or None if estimation fails.
        """
        n = signal.size
        if n < _MIN_SAMPLES_FOR_INTERPOLATION:
            return None

        # 1. Coarse search
        dft_sig = np.fft.fft(signal)
        search_space = np.abs(dft_sig[1 : n // 2])
        if search_space.size == 0:
            return None
        k_c = int(np.argmax(search_space)) + 1

        # 2. Stage 1: A&M Interpolation to get initial offset
        dfa = self._compute_am_offset(signal, k_c)

        # 3. Stage 2: QSE Refinement
        dfh = self._compute_qse_refinement(signal, k_c, dfa)

        # 4. Final frequency calculation
        refined_freq_norm = (k_c + dfh) / n
        return float(refined_freq_norm * self.fs)

    def _compute_am_offset(self, signal: ComplexArray, k_c: int) -> float:
        """Compute the initial frequency offset using A&M interpolator.

        This method implements the first stage of the two-stage HAQSE
        algorithm. It uses the Aboutanios-Mulgrew (A&M) interpolator,
        which calculates a coarse frequency offset (`dfa`) from the DTFT
        samples at `k_c ± 0.5` bins.

        Args:
            signal: The complex-valued signal to analyze.
            k_c: The coarse integer index of the spectral peak.

        Returns:
            The coarse frequency offset (`dfa`) in bins.
        """
        n = signal.size
        w_c = k_c / n

        # DTFT at k_c ± 0.5
        s_p05 = self._dtft_at(signal, w_c + 0.5 / n)
        s_m05 = self._dtft_at(signal, w_c - 0.5 / n)

        term_a = s_p05 + s_m05
        term_b = s_p05 - s_m05
        if abs(term_b) < ZERO_LEVEL:
            return 0.0

        num = np.real(term_a * np.conj(term_b))
        den = np.real(term_b * np.conj(term_b))
        return float(0.5 * (num / den))

    def _compute_qse_refinement(
        self, signal: ComplexArray, k_c: int, dfa: float
    ) -> float:
        """Compute the final frequency offset using QSE refinement.

        This method implements the second stage of the two-stage HAQSE
        algorithm. It takes the coarse offset from the A&M stage (`dfa`)
        and applies a refinement based on the Q-Shift Estimator (QSE).
        The final, refined offset is returned.

        Args:
            signal: The complex-valued signal to analyze.
            k_c: The coarse integer index of the spectral peak.
            dfa: The coarse offset from the A&M stage, in bins.

        Returns:
            The final, refined frequency offset (`dfh`) in bins,
            clipped to the range [-0.5, 0.5].
        """
        n = signal.size
        q = n ** (-1 / 3.0)

        c_q = self._compute_bias_correction(q)

        # DTFT for QSE refinement step
        w_c = k_c / n
        s_pq = self._dtft_at(signal, w_c + (dfa + q) / n)
        s_mq = self._dtft_at(signal, w_c + (dfa - q) / n)

        term_a = s_pq - s_mq
        term_b = s_pq + s_mq
        if abs(term_b) < ZERO_LEVEL or abs(c_q) < ZERO_LEVEL:
            correction = 0.0
        else:
            correction = (
                np.real(term_a * np.conj(term_b))
                / np.real(term_b * np.conj(term_b))
            ) / c_q

        dfh = dfa + correction
        return float(np.clip(dfh, -0.5, 0.5))

    @staticmethod
    def _compute_bias_correction(q: float) -> float:
        """Calculate the c(q) bias correction term for the QSE stage.

        This function computes the bias correction factor `c(q)`
        required by the Q-Shift Estimator (QSE), as defined in the
        HAQSE-family of algorithms.

        For numerical stability, it uses a Taylor series approximation
        for small values of `q`, where direct computation of `cot(pi*q)`
        can suffer from catastrophic cancellation.

        Args:
            q (float): The q-shift parameter, typically `N^(-1/3)`.

        Returns:
            float: The calculated bias correction term `c(q)`.
        """
        pi_q = np.pi * q

        if pi_q < _TAYLOR_APPROXIMATION_THRESHOLD:
            return (np.pi**2 * q) / 3.0

        with np.errstate(divide="ignore", invalid="ignore"):
            cot_val = 1.0 / np.tan(pi_q)

        if not np.isfinite(cot_val):
            return 0.0

        return float((1.0 - pi_q * cot_val) / (q * np.cos(pi_q) ** 2))

    @staticmethod
    def _dtft_at(signal: ComplexArray, freq_norm: float) -> ComplexArray:
        """Calculate the DTFT of a signal at an arbitrary frequency.

        This is a helper function that computes the Discrete-Time
        Fourier Transform (DTFT) for a given signal at a single,
        arbitrary normalized frequency point. This allows for spectral
        evaluation at off-grid frequencies.

        Args:
            signal: The complex-valued input signal.
            freq_norm: The normalized frequency (in cycles/sample) at
                which to evaluate the DTFT.

        Returns:
            The complex-valued DTFT result at the specified frequency.
        """
        n = signal.size
        t = np.arange(n)
        basis = np.exp(-2j * np.pi * freq_norm * t)
        dtft_val: ComplexArray = np.dot(signal, basis)
        return dtft_val

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

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include the interpolation
        method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["interpolator"] = self.interpolator
        return params
