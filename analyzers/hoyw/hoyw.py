# -*- coding: utf-8 -*-
"""Defines HoywAnalyzer class for Higher-Order Yule-Walker (HOYW) method.

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
from scipy.linalg import LinAlgError, svd, toeplitz
from scipy.signal import correlate

from .._common import find_freqs_from_roots
from ..base import AnalyzerBase
from ..models import AnalyzerParameters


@final
class HoywAnalyzer(AnalyzerBase):
    """Implements the Higher-Order Yule-Walker (HOYW) method."""

    def __init__(
        self, fs: float, n_sinusoids: int, *, ar_order: int | None = None
    ) -> None:
        """Initialize the HOYW analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            ar_order (int, optional): The order of the AR model.
                Should be > 2*n_sinusoids. Defaults to 512.
        """
        super().__init__(fs, n_sinusoids)
        self.ar_order = ar_order if ar_order is not None else 512

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies using the rank-truncated HOYW method.

        Args:
            signal (np.ndarray):
                Input signal (float64 or complex128).

        Returns:
            np.ndarray:
                Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
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

        # 3. Estimate AR coefficients by solving the reduced-rank HOYW
        #    equations
        ar_coeffs = self._solve_hoyw_equation(acorr_mat, acorr_vec)

        # 4. Estimate frequency by finding roots from AR coefficients
        poly_coeffs = np.concatenate(([1], ar_coeffs))
        estimated_freqs = find_freqs_from_roots(
            poly_coeffs, self.fs, self.n_sinusoids
        )

        return estimated_freqs

    @staticmethod
    def _calculate_autocorrelation(
        signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        n_lags: int,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Calculate the autocorrelation of the signal.

        Args:
            signal (np.ndarray):
                Input signal (float64 or complex128).
            n_lags (int):
                Number of autocorrelation lags to compute.

        Returns:
            np.ndarray:
                A vector of autocorrelation values
                [r(0), r(1), ..., r(n_lags-1)].
        """
        n_samples = signal.size
        corr_full = correlate(signal, signal)
        autocorr = (
            corr_full[n_samples - 1 : n_samples - 1 + n_lags] / n_samples
        )
        if np.isrealobj(signal):
            return autocorr.astype(np.float64)
        return autocorr.astype(np.complex128)

    @staticmethod
    def _build_autocorr_matrix(
        autocorr: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        p: int,
        m: int,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Build the autocorrelation matrix R for the HOYW equations.

        Args:
            autocorr (np.ndarray):
                A vector of autocorrelation values [r(0), r(1), ...].
            p (int):
                The order of the AR model.
            m (int):
                A parameter determining the number of equations, which
                should be greater than p. The total number of lags used
                is p + m.

        Returns:
            np.ndarray:
                The autocorrelation matrix R of shape (m x p)
                (float64 or complex128).
        """
        column = autocorr[p : p + m]
        row = autocorr[p:0:-1]
        acorr_mat = toeplitz(column, r=row)
        if np.isrealobj(autocorr):
            return acorr_mat.astype(np.float64)
        return acorr_mat.astype(np.complex128)

    @staticmethod
    def _build_autocorr_vector(
        autocorr: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        p: int,
        m: int,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Build the autocorrelation vector r for the HOYW equations.

        Args:
            autocorr (np.ndarray):
                A vector of autocorrelation values [r(0), r(1), ...].
            p (int):
                The order of the AR model.
            m (int):
                A parameter determining the number of equations, which
                should be greater than p. The total number of lags used
                is p + m.

        Returns:
            np.ndarray:
                The autocorrelation vector r of shape (m x 1)
                (float64 or complex128).
        """
        return autocorr[p + 1 : p + m + 1]

    def _solve_hoyw_equation(
        self,
        acorr_mat: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        acorr_vec: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Solve the HOYW equations to estimate the AR coefficients.

        Args:
            acorr_mat (np.ndarray):
                The sample autocorrelation matrix (float64 or
                complex128); lhs of Stoica 4.4.8
            acorr_vec (np.ndarray):
                The sample autocorrelation vector (float64 or
                complex128); rhs of Stoica 4.4.8

        Returns:
            np.ndarray:
                The AR coefficients (float64 or complex128).
                Returns an empty array on failure.
        """
        # Performs SVD of matrix R (Stoica 4.4.12)
        try:
            u, s, vh = svd(acorr_mat)
        except LinAlgError:
            warnings.warn("SVD on autocorrelation matrix failed.")
            return np.array([])

        # Estimate AR coefficients by solving reduced-rank equations
        #    b = -V1^H * S1_inv * U1^H * r (Stoica 4.4.16)
        #    V in textbook is Vh.conj().T
        if np.isrealobj(acorr_mat):
            # For real signals, positive and negative frequency pairs
            # are considered
            model_order = 2 * self.n_sinusoids
        else:
            # For complex signals, the number of signals themselves
            model_order = self.n_sinusoids
        u1 = u[:, :model_order]
        s1_inv = np.diag(1 / s[:model_order])
        vh1 = vh[:model_order, :]
        ar_coeffs = -vh1.conj().T @ s1_inv @ u1.conj().T @ acorr_vec
        return ar_coeffs

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters..

        Extends the base implementation to include the 'ar_order'
        parameter specific to the HOYW method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["ar_order"] = self.ar_order
        return params
