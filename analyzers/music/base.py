# -*- coding: utf-8 -*-
"""Defines MusicAnalyzerBase class for MUSIC-based parameter analyzers.

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
from abc import ABC

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from ..base import AnalyzerBase


class MusicAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for MUSIC-based parameter analyzers."""

    def _estimate_noise_subspace(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | None:
        """Estimate the noise subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated noise subspace (complex128).
                Returns None if estimation fails.
        """
        model_order = 2 * self.n_sinusoids

        # 1. Build the covariance matrix
        cov_matrix = self._build_covariance_matrix(signal, self.subspace_dim)

        # 2. Eigenvalue decomposition
        try:
            _, eigenvectors = eigh(cov_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Eigenvalue decomposition on covariance matrix failed.")
            return None

        # The noise subspace is the set of vectors corresponding to the smaller
        # eigenvalues.
        # Since it is in ascending order, select (subspace_dim - model_order) vectors
        # from the beginning
        n_noise_vectors = self.subspace_dim - model_order
        _subspace = eigenvectors[:, :n_noise_vectors]
        noise_subspace: npt.NDArray[np.complex128] = _subspace.astype(np.complex128)
        return noise_subspace
