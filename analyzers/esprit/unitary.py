# -*- coding: utf-8 -*-
"""Defines UnitaryEspritAnalyzer class for Unitary ESPRIT.

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
from scipy.linalg import eigh, hankel

from .base import EspritAnalyzerBase
from .solvers import LSUnitaryEspritSolver, TLSUnitaryEspritSolver


@final
class UnitaryEspritAnalyzer(EspritAnalyzerBase):
    """A class to solve frequencies via Unitary ESPRIT with least squares."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSUnitaryEspritSolver | LSUnitaryEspritSolver,
        sep_factor: float = 0.4,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSUnitaryEspritSolver | TLSUnitaryEspritSolver):
                Solver to solve frequencies with the rotation operator.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
        """
        super().__init__(fs, n_sinusoids)
        self.solver: LSUnitaryEspritSolver | TLSUnitaryEspritSolver = solver
        self.sep_factor: float = sep_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Estimate the signal subspace
        signal_subspace = self._estimate_signal_subspace(signal)
        if signal_subspace is None:
            return np.array([])

        # 2. Solve frequencies with the stored solver
        omegas = self.solver.solve(signal_subspace, self.subspace_dim)

        # 3. Convert normalized angular frequencies [rad/sample]
        #    to physical frequencies [Hz]
        estimated_freqs_hz = omegas * (self.fs / (2 * np.pi))

        # 4. Extract and sort only pairs with positive frequencies
        positive_freq_indices = np.where(estimated_freqs_hz > 0)[0]
        sorted_indices = np.argsort(estimated_freqs_hz[positive_freq_indices])
        raw_freqs = estimated_freqs_hz[positive_freq_indices][sorted_indices]

        # 5. Filter unique frequencies
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        est_freqs = self._filter_unique_freqs(raw_freqs, min_separation_hz)

        return est_freqs

    @override
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64] | None:
        """Estimate the real-valued signal subspace using the covariance approach.

        This method follows Step 2 of TABLE I in Haardt & Nossek (1995),
        which involves a real-valued SVD of a transformed data matrix.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated signal subspace matrix (complex128).
                Returns None if estimation fails.
        """
        # 1. Construct the data matrix X (Hankel matrix)
        #    size: (L, N) = (subspace_dim, n_snapshots)
        data_matrix = hankel(
            signal[: self.subspace_dim], signal[self.subspace_dim - 1 :]
        )

        # 2. Convert complex matrix X to real matrix T(X) (based on Eq. (7))
        #    The size of T(X) is (L, 2*N)
        try:
            transformed_matrix = self._transform_complex_to_real(data_matrix)
        except ValueError:
            warnings.warn("Failed to transform complex data matrix to real matrix.")
            return None

        # 3. Perform eigenvalue decomposion the Hermetial matrix in Eq. (29)
        cov_matrix = transformed_matrix @ transformed_matrix.conj().T
        _, eigenvectors = eigh(cov_matrix)

        # 4. Estimated signal subspace is the 2*M principal eigenvectors
        signal_subspace = eigenvectors[:, -2 * self.n_sinusoids :]
        return signal_subspace.astype(np.float64)

    @staticmethod
    def _transform_complex_to_real(
        g: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
        """Transform a complex matrix G to a real matrix T(G) based on Eq. (7).

        Args:
            g (np.ndarray): Complex matrix G (complex128).

        Returns:
            np.ndarray: Transformed real matrix T(G) (complex128).
        """
        p, _ = g.shape
        p_half = p // 2  # L

        if p % 2 == 0:  # L is even
            g1 = g[:p_half, :]
            g2 = g[p_half:, :]
            pi_g2 = np.flipud(g2.conj())
            _sum = g1 + pi_g2
            _diff = g1 - pi_g2
            tg_left = np.vstack([np.real(_sum), np.imag(_sum)])
            tg_right = np.vstack([-np.imag(_diff), np.real(_diff)])
            return np.hstack([tg_left, tg_right])

        # L is odd
        g1 = g[:p_half, :]
        gt = g[p_half, :]
        g2 = g[p_half + 1 :, :]
        pi_g2 = np.flipud(g2.conj())
        _sum = g1 + pi_g2
        _diff = g1 - pi_g2
        tg_left = np.vstack([np.real(_sum), np.sqrt(2.0) * np.real(gt), np.imag(_sum)])
        tg_right = np.vstack(
            [-np.imag(_diff), -np.sqrt(2.0) * np.imag(gt), np.real(_diff)]
        )
        return np.hstack([tg_left, tg_right])
