# -*- coding: utf-8 -*-
"""Defines EspritAnalyzer class for ESPRIT using Least Squares.

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
from typing import override

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvals, pinv

from .base import EspritAnalyzerBase


class LSEspritAnalyzer(EspritAnalyzerBase):
    """A class to solve frequencies via using Least Squares."""

    @override
    def _solve_params_from_subspace(
        self, signal_subspace: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Solve for frequencies from the signal subspace."""
        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]
        try:
            rotation_operator_psi = pinv(subspace_upper) @ subspace_lower
        except np.linalg.LinAlgError:
            warnings.warn("Matrix inversion failed in parameter solving.")
            return np.array([])
        try:
            eigenvalues_psi = eigvals(rotation_operator_psi)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Eigenvalue decomposition failed while solving rotation operator."
            )
            return np.array([])
        angles = np.angle(eigenvalues_psi)
        estimated_freqs_hz = angles * (self.fs / (2 * np.pi))

        # Extract and sort only pairs with positive frequencies
        positive_freq_indices = np.where(estimated_freqs_hz > 0)[0]

        sorted_indices = np.argsort(estimated_freqs_hz[positive_freq_indices])
        freqs = estimated_freqs_hz[positive_freq_indices][sorted_indices]
        return freqs.astype(np.float64)
