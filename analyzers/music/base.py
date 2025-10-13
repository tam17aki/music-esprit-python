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

from utils.data_models import ComplexArray, FloatArray, SignalArray

from .._common import compute_subspace_from_cov
from ..base import AnalyzerBase


class MusicAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for MUSIC-based parameter analyzers."""

    def _estimate_noise_subspace(
        self, signal: SignalArray
    ) -> FloatArray | ComplexArray | None:
        """Estimate the noise subspace using eigenvalue decomposition.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            FloatArray | ComplexArray | None: The noise subspace matrix.
                Returns None on failure.
        """
        # Build the covariance matrix
        cov_matrix = self._build_covariance_matrix(signal, self.subspace_dim)

        # The noise subspace is the set of vectors corresponding to the
        # smaller eigenvalues.  Since it is in ascending order, select
        # (subspace_dim - model_order) vectors from the beginning, where
        # model_order = 2 * n_sinusoids if signal is real-valued, else
        # model_order = n_sinusoids if signal is complex-valued.
        if np.isrealobj(signal):
            # For real signals, positive and negative frequency pairs
            # are considered
            model_order = 2 * self.n_sinusoids
        else:
            # For complex signals, the number of signals themselves
            model_order = self.n_sinusoids

        # Estimate the noise subspace using eigenvalue decomposition
        _, noise_subspace = compute_subspace_from_cov(cov_matrix, model_order)
        if noise_subspace is None:
            warnings.warn("EVD on covariance matrix failed in Spectral MUSIC.")
            return None
        return noise_subspace
