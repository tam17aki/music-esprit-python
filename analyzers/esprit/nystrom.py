# -*- coding: utf-8 -*-
"""Defines NystromEspritAnalyzer class to solve frequencies via Nyström-ESPRIT.

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
from scipy.linalg import eigh, qr

from ..models import AnalyzerParameters
from .base import EVDBasedEspritAnalyzer
from .solvers import LSEspritSolver, TLSEspritSolver


@final
class NystromEspritAnalyzer(EVDBasedEspritAnalyzer):
    """A class to solve frequencies via Nyström-ESPRIT."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSEspritSolver | TLSEspritSolver,
        *,
        nystrom_rank_factor: int = 10,
    ) -> None:
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSEspritSolver | TLSEspritSolver):
                Solver to solve frequencies with the rotation operator.
            nystrom_rank_factor (int, optional):
                A factor to determine the number of rows to sample for the
                Nyström approximation (P = factor * 2M). A larger value
                improves robustness at the cost of computation. Defaults to 10.
        """
        super().__init__(fs, n_sinusoids)
        self.solver: LSEspritSolver | TLSEspritSolver = solver
        self.nystrom_rank_factor = nystrom_rank_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Estimate the signal subspace
        signal_subspace = self._estimate_signal_subspace(signal)
        if signal_subspace is None:
            return np.array([])

        # 2. Solve frequencies with the stored solver
        omegas = self.solver.solve(signal_subspace)

        # 3. Post-processes raw angular frequencies to final frequency estimates
        est_freqs = self._postprocess_omegas(omegas)

        return est_freqs

    @override
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None:
        """Approximates the signal subspace using the Nyström method.

        This method avoids the computationally expensive EVD of the full
        covariance matrix. Instead, it constructs an approximation of the
        signal subspace by performing EVD on smaller, sampled sub-matrices
        (R11 and F^H*F), significantly reducing computational complexity.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray:
                An orthonormal basis for the approximated signal subspace
                (complex128). Returns None if estimation fails.
        """
        # --- Step 1: Prepare the parameters ---
        model_order = self.nystrom_rank_factor * self.n_sinusoids * 2
        if model_order >= self.subspace_dim:
            warnings.warn("Nystrom rank is too large for the subspace dimension.")
            model_order = self.subspace_dim - 1

        data_matrix = self._build_hankel_matrix(signal, self.subspace_dim)

        # --- Step 2: Calculate R11 and R12 ---
        r11, r12 = self._compute_sub_covariance_matrices(data_matrix, model_order)

        # --- Step 3: Build the matrix F ---
        try:
            matrix_f = self._build_f_matrix(r11, r12)
        except np.linalg.LinAlgError:
            warnings.warn("Failed to build the F matrix in Nystrom method.")
            return None

        # --- Step 4: Compute the signal subspace from F ---
        try:
            signal_subspace = self._compute_subspace_from_f(matrix_f)
        except np.linalg.LinAlgError:
            warnings.warn("EVD of F^H*F failed in Nystrom method.")
            return None

        return signal_subspace

    @staticmethod
    def _compute_sub_covariance_matrices(
        data_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        model_order: int,
    ) -> tuple[
        npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ]:
        """Computes the R11 and R12 sub-matrices of the covariance matrix."""
        x1 = data_matrix[:model_order, :]
        x2 = data_matrix[model_order:, :]
        n_snapshots = data_matrix.shape[1]
        r11 = (x1 @ x1.conj().T) / n_snapshots
        r12 = (x1 @ x2.conj().T) / n_snapshots
        return r11, r12

    def _build_f_matrix(
        self,
        r11: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        r12: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Builds the intermediate matrix F based on Eq. (12)."""
        eigvals_r11, u_a = eigh(r11)
        safe_eigvals = np.maximum(eigvals_r11, 1e-12)
        l_a_inv_sqrt_diag = 1 / np.sqrt(safe_eigvals)
        r11_inv_sqrt = u_a @ np.diag(l_a_inv_sqrt_diag) @ u_a.conj().T

        f_block = np.vstack([r11, r12.conj().T])
        _f_matrix = f_block @ r11_inv_sqrt
        if np.isrealobj(_f_matrix):
            f_matrix_float: npt.NDArray[np.float64] = _f_matrix.astype(np.float64)
            return f_matrix_float
        f_matrix_complex: npt.NDArray[np.complex128] = _f_matrix.astype(np.complex128)
        return f_matrix_complex

    def _compute_subspace_from_f(
        self, matrix_f: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Computes the signal subspace from the F matrix based on Eq. (11, 13)."""
        f_h_f = matrix_f.conj().T @ matrix_f
        eigvals_f, u_f = eigh(f_h_f)

        safe_eigvals_f = np.maximum(eigvals_f, 1e-12)
        l_f_inv_sqrt = np.diag(1 / np.sqrt(safe_eigvals_f))

        signal_subspace = matrix_f @ u_f @ l_f_inv_sqrt

        _q_matrix, _ = qr(signal_subspace, mode="economic")
        if np.isrealobj(_q_matrix):
            q_matrix_float: npt.NDArray[np.float64] = _q_matrix.astype(np.float64)
            return q_matrix_float
        q_matrix_complex: npt.NDArray[np.complex128] = _q_matrix.astype(np.complex128)
        return q_matrix_complex

    @override
    def get_params(self) -> AnalyzerParameters:
        """Returns the analyzer's hyperparameters.

        Extends the base implementation to include the name of the solver class.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and specific hyperparameters.
        """
        params = super().get_params()
        params["solver"] = self.solver.__class__.__name__
        return params
