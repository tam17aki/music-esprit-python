# -*- coding: utf-8 -*-
"""Defines StandardEspritAnalyzer class to solve frequencies via ESPRIT.

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
from scipy.linalg import LinAlgError, eigh

from mixins.covariance import ForwardBackwardMixin

from .base import EspritAnalyzerBase
from .solvers import LSEspritSolver, TLSEspritSolver


class StandardEspritAnalyzer(EspritAnalyzerBase):
    """A class to solve frequencies via ESPRIT."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSEspritSolver | TLSEspritSolver,
        subspace_ratio: float = 1 / 3,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSEspritSolver | TLSEspritSolver):
                Solver to solve frequencies with the rotation operator.
            subspace_ratio (float, optional): The ratio of the subspace dimension
                to the signal length. Must be between 0 and 0.5. Defaults to 1/3.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio)
        self.solver: LSEspritSolver | TLSEspritSolver = solver

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
        omegas = self.solver.solve(signal_subspace)

        # 3. Post-processes raw angular frequencies to final frequency estimates
        est_freqs = self._postprocess_omegas(omegas)

        return est_freqs

    @override
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | None:
        """Estimate the signal subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: The complex-valued signal subspace matrix (complex128).
                Returns None if estimation fails.
        """
        cov_matrix = self._build_covariance_matrix(signal, self.subspace_dim)
        try:
            _, eigenvectors = eigh(cov_matrix)
        except LinAlgError:
            warnings.warn("Eigenvalue decomposition on covariance matrix failed.")
            return None

        # Estimated signal subspace is the 2*M principal eigenvectors
        signal_subspace = eigenvectors[:, -2 * self.n_sinusoids :].astype(np.complex128)
        return signal_subspace


@final
class StandardEspritAnalyzerFB(ForwardBackwardMixin, StandardEspritAnalyzer):
    """ESPRIT analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
