# -*- coding: utf-8 -*-
"""Defines EspritAnalyzer class for ESPRIT using Total Least Squares.

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
from scipy.linalg import eigvals, pinv, svd

from mixins.covariance import ForwardBackwardMixin

from .base import EspritAnalyzerBase


class TLSEspritAnalyzer(EspritAnalyzerBase):
    """A class to solve frequencies via ESPRIT using Total Least Squares."""

    @staticmethod
    def _compute_rotation_operator_tls(
        signal_subspace: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128] | None:
        """Compute the rotation operator Psi using Total Least Squares (TLS).

        Args:
            signal_subspace (np.ndarray): The signal subspace matrix (complex128).

        Returns:
            np.ndarray: The rotation operator matrix (complex128).
                Returns None if estimation fails.
        """
        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]

        # Form the augmented matrix for SVD
        augmented_subspace = np.concatenate((subspace_upper, subspace_lower), axis=1)
        try:
            _, _, vh = svd(augmented_subspace)
        except np.linalg.LinAlgError:
            warnings.warn("SVD on augmented_subspace did not converge.")
            return None

        # Partition the Vh matrix to solve the TLS problem
        model_order = signal_subspace.shape[1]
        v12 = vh[model_order:, :model_order]
        v22 = vh[model_order:, model_order:]

        # Solve the rotation operator
        try:
            rotation_operator = -v12 @ pinv(v22)
        except np.linalg.LinAlgError:
            warnings.warn(
                "TLS matrix inversion failed while computing rotation operator."
            )
            return None

        return rotation_operator

    @override
    def _solve_freqs_from_subspace(
        self, signal_subspace: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Solve for frequencies from the signal subspace.

        Args:
            signal_subspace (np.ndarray): The signal subspace matrix (complex128).

        Returns:
            np.ndarray: Estimated frequencies (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Compute the rotation operator Psi using TLS
        rotation_operator_psi = self._compute_rotation_operator_tls(signal_subspace)
        if rotation_operator_psi is None:
            return np.array([])

        # 2. Compute eigenvalues of Psi
        try:
            eigenvalues_psi = eigvals(rotation_operator_psi)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Eigenvalue decomposition failed while solving rotation operator."
            )
            return np.array([])

        # 3. Estimate normalized angular frequencies from the eigenvalues
        angles = np.angle(eigenvalues_psi)

        # 4. Convert normalized angular frequencies [rad/sample]
        #    to physical frequencies [Hz]
        estimated_freqs_hz = angles * (self.fs / (2 * np.pi))

        # 5. Extract and sort only pairs with positive frequencies
        positive_freq_indices = np.where(estimated_freqs_hz > 0)[0]
        sorted_indices = np.argsort(estimated_freqs_hz[positive_freq_indices])
        freqs = estimated_freqs_hz[positive_freq_indices][sorted_indices]

        return freqs.astype(np.float64)


@final
class TLSEspritAnalyzerFB(ForwardBackwardMixin, TLSEspritAnalyzer):
    """ESPRIT-TLS analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
