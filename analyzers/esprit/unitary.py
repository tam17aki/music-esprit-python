# -*- coding: utf-8 -*-
"""Defines UnitaryEspritAnalyzer class to solve frequencies via Unitary ESPRIT.

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
from scipy.linalg import LinAlgError, eigh, hankel

from ..models import AnalyzerParams
from .base import EspritAnalyzerBase
from .solvers import LSUnitaryEspritSolver, TLSUnitaryEspritSolver


@final
class UnitaryEspritAnalyzer(EspritAnalyzerBase):
    """A class to solve frequencies via Unitary ESPRIT with least squares."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSUnitaryEspritSolver | TLSUnitaryEspritSolver,
        *,
        subspace_ratio: float = 1 / 3,
    ) -> None:
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSUnitaryEspritSolver | TLSUnitaryEspritSolver):
                Solver to solve frequencies with the rotation operator.
            subspace_ratio (float, optional): The ratio of the subspace dimension
                to the signal length. Must be between 0 and 0.5. Defaults to 1/3.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio)
        self.solver: LSUnitaryEspritSolver | TLSUnitaryEspritSolver = solver

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
    ) -> npt.NDArray[np.float64] | None:
        """Estimate the real-valued signal subspace using the covariance approach.

        This method follows Step 2 of TABLE I in Haardt & Nossek (1995),
        which involves a real-valued SVD of a transformed data matrix.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: The real-valued signal subspace matrix (float64).
                Returns None if estimation fails.
        """
        # 1. Construct the data matrix X (Hankel matrix)
        #    size: (L, N) = (subspace_dim, n_snapshots)
        _data_matrix = hankel(
            signal[: self.subspace_dim], signal[self.subspace_dim - 1 :]
        )
        data_matrix = _data_matrix.astype(np.complex128)

        # 2. Convert complex matrix X to real matrix T(X) (based on Eq. (7))
        #    The size of T(X) is (L, 2*N)
        try:
            transformed_matrix = self._transform_complex_to_real(data_matrix)
        except ValueError:
            warnings.warn("Failed to transform complex data matrix to real matrix.")
            return None

        # 3. Perform eigenvalue decomposion the Hermetial matrix in Eq. (29)
        cov_matrix = transformed_matrix @ transformed_matrix.conj().T
        try:
            _, eigenvectors = eigh(cov_matrix)
        except LinAlgError:
            warnings.warn("Eigenvalue decomposition on covariance matrix failed.")
            return None

        # 4. Estimated signal subspace is the 2*M principal eigenvectors
        signal_subspace = eigenvectors[:, -2 * self.n_sinusoids :]

        return signal_subspace

    @staticmethod
    def _transform_complex_to_real(
        g_matrix: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
        """Transform a complex matrix G to a real matrix T(G) based on Eq. (7).

        Args:
            g_matrix (np.ndarray): Complex matrix G (complex128).

        Returns:
            np.ndarray: Transformed real matrix T(G) (float64).
        """
        p, _ = g_matrix.shape
        p_half = p // 2  # L

        if p % 2 == 0:  # L is even
            g1 = g_matrix[:p_half, :]
            g2 = g_matrix[p_half:, :]
            pi_g2 = np.flipud(g2.conj())
            _sum = g1 + pi_g2
            _diff = g1 - pi_g2
            tg_left = np.vstack([np.real(_sum), np.imag(_sum)])
            tg_right = np.vstack([-np.imag(_diff), np.real(_diff)])
        else:  # L is odd
            g1 = g_matrix[:p_half, :]
            gt = g_matrix[p_half, :]
            g2 = g_matrix[p_half + 1 :, :]
            pi_g2 = np.flipud(g2.conj())
            _sum = g1 + pi_g2
            _diff = g1 - pi_g2
            tg_left = np.vstack(
                [np.real(_sum), np.sqrt(2.0) * np.real(gt), np.imag(_sum)]
            )
            tg_right = np.vstack(
                [-np.imag(_diff), -np.sqrt(2.0) * np.imag(gt), np.real(_diff)]
            )

        return np.hstack([tg_left, tg_right]).astype(np.float64)

    @override
    def get_params(self) -> AnalyzerParams:
        """Returns the analyzer's hyperparameters, including spectral-specific ones.

        Extends the base implementation to include the name of the solver class.

        Returns:
            AnalyzerParams:
                A TypedDict containing both common and spectral-specific
                hyperparameters.
        """
        params = super().get_params()
        params["solver"] = self.solver.__class__.__name__
        return params
