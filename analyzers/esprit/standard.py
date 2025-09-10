# -*- coding: utf-8 -*-
"""Defines LSEspritAnalyzer class for ESPRIT using Least Squares.

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
from scipy.linalg import eigh

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
        sep_factor: float = 0.4,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSEspritSolver | TLSEspritSolver):
                Solver to solve frequencies with the rotation operator.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
        """
        super().__init__(fs, n_sinusoids)
        self.solver: LSEspritSolver | TLSEspritSolver = solver
        self.sep_factor: float = sep_factor

    @override
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | None:
        """Estimate the signal subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated signal subspace matrix (complex128).
                Returns None if estimation fails.
        """
        cov_matrix = self._build_covariance_matrix(signal, self.subspace_dim)
        try:
            _, eigenvectors = eigh(cov_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Eigenvalue decomposition on covariance matrix failed.")
            return None
        _subspace = eigenvectors[:, -2 * self.n_sinusoids :]
        signal_subspace: npt.NDArray[np.complex128] = _subspace.astype(np.complex128)
        return signal_subspace

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

        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]

        # 2. Solve frequencies with the stored solver
        omegas = self.solver.solve(subspace_upper, subspace_lower)

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


@final
class LSEspritAnalyzerFB(ForwardBackwardMixin, StandardEspritAnalyzer):
    """ESPRIT analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
