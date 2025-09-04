# -*- coding: utf-8 -*-
"""Defines MinNormAnalyzerBase class for MinNorm-based parameter analyzers.

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

from ..music.base import MusicAnalyzerBase


class MinNormAnalyzerBase(MusicAnalyzerBase, ABC):
    """Abstract base class for MinNorm-based parameter analyzers."""

    def _calculate_min_norm_vector(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | None:
        """Calculate the minimum norm vector 'd' from the noise subspace E_n."""
        # Estimate the noise subspace (reusing the base class method)
        noise_subspace = self._estimate_noise_subspace(signal)
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace.")
            return None

        # Extract the first row vector of the noise subspace E_n
        first_row_h = noise_subspace[0, :]

        # If the first row vector is close to the zero vector,
        # the calculation may become unstable.
        if np.linalg.norm(first_row_h) < 1e-9:
            warnings.warn("The first row of the noise subspace is close to zero.")
            return None

        # d is a linear combination of the column vectors of E_n, d = E_n * w
        # The constraint that the first element is 1 can be written as e1^H * d = 1
        # e1 is [1, 0, ..., 0]^T
        # (first row vector of E_n) * w = 1
        # w = (first row of E_n)^H / ||first row of E_n||^2

        # Calculate the coupling coefficient w
        weights = first_row_h.conj().T / (first_row_h @ first_row_h.conj().T)

        # Calculate the minimum norm vector d
        min_norm_vector = noise_subspace @ weights

        return min_norm_vector
