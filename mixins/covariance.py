# -*- coding: utf-8 -*-
"""A mixin class providing Forward-Backward averaging for covariance matrix estimation.

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

import numpy as np
import numpy.typing as npt
from scipy.linalg import hankel


class ForwardBackwardMixin:
    """A mixin class providing Forward-Backward averaging for cov. matrix estimation.

    This mixin overrides the `_build_covariance_matrix` method to enhance estimation
    accuracy, especially for short data records or low SNR scenarios.
    """

    @staticmethod
    def _build_covariance_matrix(
        signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128], subspace_dim: int
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Build the forward-backward averaged covariance matrix.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).
            subspace_dim (int): The dimension of subspace.

        Returns:
            np.ndarray:
                The forward-backward averaged covariance matrix (float64 or complex128).
        """
        # 1. Standard forward covariance matrix
        n_samples = signal.size
        n_snapshots = n_samples - subspace_dim + 1
        hankel_matrix = hankel(signal[:subspace_dim], signal[subspace_dim - 1 :])
        cov_matrix_f = (hankel_matrix @ hankel_matrix.conj().T) / n_snapshots

        # 2. Backward covariance matrix
        exchange_matrix = np.ascontiguousarray(np.fliplr(np.eye(subspace_dim)))
        cov_matrix_b = exchange_matrix @ cov_matrix_f.conj() @ exchange_matrix

        # 3. Averaged covariance matrix
        cov_matrix_fb = (cov_matrix_f + cov_matrix_b) / 2.0
        if np.isdtype(np.float64, signal.dtype):
            return cov_matrix_fb.astype(np.float64)
        return cov_matrix_fb.astype(np.complex128)
