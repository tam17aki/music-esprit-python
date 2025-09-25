# -*- coding: utf-8 -*-
"""Defines FFTEspritAnalyzer class for FFT-ESPRIT algorithm.

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
from scipy.linalg import qr

from .._common import estimate_freqs_iterative_fft
from ..models import AnalyzerParameters
from .base import EspritAnalyzerBase
from .solvers import LSEspritSolver, TLSEspritSolver


@final
class FFTEspritAnalyzer(EspritAnalyzerBase):
    """Analyzes sinusoidal parameters using the FFT-ESPRIT algorithm.

    This analyzer provides a computationally efficient alternative to the
    standard ESPRIT method. Instead of a full SVD/EVD on a covariance
    matrix, it approximates the signal subspace using a kernel-based
    approach leveraging the Fast Fourier Transform (FFT).

    The process involves:
    1. A rough, off-grid estimation of frequencies using an iterative
       interpolated FFT method (similar to IIp-DFT).
    2. Construction of a truncated DFT kernel (a Vandermonde-like matrix)
       based on these rough frequency estimates.
    3. Projecting the data onto this kernel to get a "cleaned" data matrix.
    4. Estimating the signal subspace from this cleaned matrix via QR decomposition.
    5. Applying the standard ESPRIT rotational invariance technique.

    This method can achieve quasi-linear time complexity O(N log N) and
    can be more robust than standard ESPRIT in very low SNR regimes.

    Reference:
        S. L. Kiser, et al., "Fast Kernel-based Signal Subspace Estimates for
        Line Spectral Estimation," PREPRINT, 2023.
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSEspritSolver | TLSEspritSolver,
        *,
        n_fft_iip: int | None = None,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSEspritSolver | TLSEspritSolver):
                Solver to solve frequencies with the rotation operator.
            n_fft_iip (int): The length of iterative interpolation FFT.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio=0.5)
        self.solver = solver
        self.n_fft_iip = n_fft_iip

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate the signal subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Build the data matrix X (Hankel matrix)
        data_matrix = self._build_hankel_matrix(signal, self.subspace_dim)

        # 2. Obtain a rough estimate of the frequency using an IIp-DFT-like
        #    method (Alg. 2, Step 2 of the paper)
        rough_freqs = estimate_freqs_iterative_fft(
            signal, self.n_sinusoids, self.fs, self.n_fft_iip
        )
        if rough_freqs.size == 0:
            warnings.warn("Initial frequency estimation (IIp-DFT) failed.")
            return np.array([])

        if np.isrealobj(signal):
            # For real signals, consider positive and negative frequency pairs
            kernel_freqs = np.concatenate([rough_freqs, -rough_freqs])
        else:
            # For complex signals, use only estimated frequencies
            kernel_freqs = -rough_freqs

        # 3. Build a truncated DFT kernel (A_p_hat) from the coarse frequency
        #    estimate (Step 3)
        n_snapshots = signal.size - self.subspace_dim + 1
        kernel_matrix = self._build_vandermonde_matrix(
            kernel_freqs, n_snapshots, self.fs
        )

        # 4. Project the data onto a matrix to obtain an approximate signal space
        #    matrix Yp (Step 4)
        projected_matrix = data_matrix @ kernel_matrix

        # 5. Obtain an orthonormal signal subspace Q from the QR decomposition of
        #    Yp (Step 5)
        try:
            # Performs efficient thin QR decomposition with "economic" mode
            _q_matrix, _ = qr(projected_matrix, mode="economic")
            q_matrix = _q_matrix.astype(np.complex128)
        except np.linalg.LinAlgError:
            warnings.warn("QR decomposition failed in FFT-ESPRIT.")
            return np.array([])

        # 6. Apply the standard ESPRIT solver to Q (Steps 6-8)
        omegas = self.solver.solve(q_matrix)

        # 7. Post-processing
        est_freqs = self._postprocess_omegas(omegas)
        return est_freqs

    @override
    def get_params(self) -> AnalyzerParameters:
        """Returns the analyzer's hyperparameters.

        Extends the base implementation to include the name of the solver class.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and specific hyperparameters.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["solver"] = self.solver.__class__.__name__
        params["n_fft_iip"] = self.n_fft_iip
        return params
