# -*- coding: utf-8 -*-
"""Defines HOYWAnalyzer class for Higher-Order Yule-Walker (HOYW) method.

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
from scipy.linalg import svd, toeplitz
from scipy.signal import correlate

from .._common import find_freqs_from_roots
from ..base import AnalyzerBase


@final
class HOYWAnalyzer(AnalyzerBase):
    """Parameter analyzer using the Higher-Order Yule-Walker (HOYW) method."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        ar_order: int | None = None,
        sep_factor: float = 0.4,
    ):
        """Initialize the HOYW analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            ar_order (int, optional): The order of the AR model.
                Should be > 2*n_sinusoids. Defaults to 128*n_sinusoids.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
        """
        super().__init__(fs, n_sinusoids)
        self.ar_order = ar_order if ar_order is not None else 128 * self.n_sinusoids
        self.sep_factor = sep_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies using the rank-truncated HOYW method.

        Args:
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        model_order = 2 * self.n_sinusoids  # n in the textbook
        p = self.ar_order  # L in the textbook
        m = p  # M in the textbook
        n_lags = p + m + 1  # Desired number of lags for autocorrelation
        if signal.size < n_lags:
            warnings.warn(
                "Signal is too short for HOYW method. "
                + "The order of the AR model (ar_order) maybe too high."
            )
            return np.array([])

        # 1. Calculate the autocorrelation
        autocorr = self._calculate_autocorrelation(signal, n_lags)

        # 2. Construct the matrix R and vector r of the HOYW equation
        try:
            acorr_mat = self._build_autocorr_matrix(autocorr, p, m)  # m x p
            acorr_vec = self._build_autocorr_vector(autocorr, p, m)  # m x 1
        except ValueError as e:
            warnings.warn(f"Failed to build Yule-Walker matrices: {e}")
            return np.array([])

        # 3. Estimate AR coefficients by solving the reduced-rank HOYW equations
        ar_coeffs = self._solve_hoyw_equation(acorr_mat, acorr_vec, model_order)

        # 4. Estimate frequency by finding roots from AR coefficients
        poly_coeffs = np.concatenate(([1], ar_coeffs))
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        estimated_freqs = find_freqs_from_roots(
            self.fs, self.n_sinusoids, poly_coeffs, min_separation_hz
        )

        return estimated_freqs

    @staticmethod
    def _calculate_autocorrelation(
        signal: npt.NDArray[np.complex128], n_lags: int
    ) -> npt.NDArray[np.complex128]:
        """Calculate the autocorrelation of the signal.

        Args:
            signal (np.ndarray):
                Input signal frame (complex128).
            n_lags (int):
                The number of autocorrelation lags to compute (including lag 0).

        Returns:
            np.ndarray:
                A vector of autocorrelation values [r(0), r(1), ..., r(num_lags-1)].
        """
        n_samples = signal.size
        corr_full = correlate(signal, signal)
        return corr_full[n_samples - 1 : n_samples - 1 + n_lags] / n_samples

    @staticmethod
    def _build_autocorr_matrix(
        autocorr: npt.NDArray[np.complex128], p: int, m: int
    ) -> npt.NDArray[np.complex128]:
        """Build the autocorrelation matrix R for the HOYW equations.

        Args:
            autocorr (np.ndarray):
                A vector of autocorrelation values [r(0), r(1), ...].
            p (int):
                The order of the AR model.
            m (int):
                A parameter determining the number of equations, which should
                be greater than p. The total number of lags used is p + m.

        Returns:
            np.ndarray:
                The (m x p) autocorrelation matrix R.
        """
        column = autocorr[p : p + m]
        row = autocorr[p:0:-1]
        return toeplitz(column, r=row)

    @staticmethod
    def _build_autocorr_vector(
        autocorr: npt.NDArray[np.complex128], p: int, m: int
    ) -> npt.NDArray[np.complex128]:
        """Build the autocorrelation vector r for the HOYW equations.

        Args:
            autocorr (np.ndarray):
                A vector of autocorrelation values [r(0), r(1), ...].
            p (int):
                The order of the AR model.
            m (int):
                A parameter determining the number of equations, which should
                be greater than p. The total number of lags used is p + m.

        Returns:
            np.ndarray:
                The (m x 1) autocorrelation vector r.
        """
        return autocorr[p + 1 : p + m + 1]

    @staticmethod
    def _solve_hoyw_equation(
        acorr_mat: npt.NDArray[np.complex128],
        acorr_vec: npt.NDArray[np.complex128],
        model_order: int,
    ) -> npt.NDArray[np.complex128]:
        """Solve the reduced-rank HOYW equations to estimate the AR coefficients.

        Args:
            acorr_mat (np.ndarray):
                The sample autocorrelation matrix (float64); lhs of Stoica 4.4.8
            acorr_vec (np.ndarray):
                The sample autocorrelation vector (float64); rhs of Stoica 4.4.8
            model_order (int):
                The order of model; model_order = 2 * n_sinusoids

        Returns:
            np.ndarray: The AR coefficients.
                Returns empty arrays if estimation fails.
        """
        # Performs SVD of matrix R (Stoica 4.4.12)
        try:
            u, s, vh = svd(acorr_mat)
        except np.linalg.LinAlgError:
            warnings.warn("SVD on autocorrelation matrix failed.")
            return np.array([])

        # Estimate AR coefficients by solving reduced-rank equations
        #    b = -V1^H * S1_inv * U1^H * r (Stoica 4.4.16)
        #    V in textbook is Vh.conj().T
        u1 = u[:, :model_order]
        s1_inv = np.diag(1 / s[:model_order])
        vh1 = vh[:model_order, :]
        ar_coeffs = -vh1.conj().T @ s1_inv @ u1.conj().T @ acorr_vec

        return ar_coeffs
