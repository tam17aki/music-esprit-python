# -*- coding: utf-8 -*-
"""Defines FFTEspritAnalyzer class for Fast FFT-ESPRIT algorithm.

Copyright (C) 2025 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import warnings
from typing import final, override

import numpy as np
import numpy.typing as npt
from scipy.linalg import qr
from scipy.signal import fftconvolve

from .._common import estimate_freqs_iterative_fft
from ..models import AnalyzerParameters
from .base import EspritAnalyzerBase
from .solvers import LSEspritSolver, TLSEspritSolver, WoodburyLSEspritSolver


@final
class FFTEspritAnalyzer(EspritAnalyzerBase):
    """Analyzes sinusoidal parameters via the Fast FFT-ESPRIT algorithm.

    This analyzer provides a computationally efficient alternative to
    standard ESPRIT, achieving a quasi-linear time complexity of O(N log
    N). Instead of a full SVD/EVD, it approximates the signal subspace
    using a kernel-based approach that leverages the Fast Fourier
    Transform (FFT).

    The process involves:

    1. A rough, off-grid estimation of frequencies via an iterative FFT
       method.
    2. Construction of a truncated DFT kernel from these rough
       estimates.
    3. An efficient projection of the data onto this kernel using
       FFT-based fast convolution (the "Fast Hankel Matrix-Matrix
       product").
    4. Estimation of the signal subspace via QR decomposition.
    5. Application of a standard ESPRIT solver to the approximated
       subspace.

    This method is particularly suitable for long signals or real-time
    applications and can be more robust than standard ESPRIT in very low
    SNR regimes.

    Reference:
        S. L. Kiser, et al., "Fast Kernel-based Signal Subspace
        Estimates for Line Spectral Estimation," PREPRINT,
        2023. (Specifically, Algorithm 4)
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: LSEspritSolver | TLSEspritSolver | WoodburyLSEspritSolver,
        *,
        n_fft_iip: int | None = None,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (LSEspritSolver | TLSEspritSolver | WoodburyLSEspritSolver):
                Solver to solve frequencies with the rotation operator.
            n_fft_iip (int): The length of iterative interpolation FFT.
        """
        super().__init__(fs, n_sinusoids)
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
        # 1. Obtain a rough estimate of the frequency using an
        #    IIp-DFT-like method
        #    (Corresponds to Alg. 4, Step 1 of the paper)
        rough_freqs = estimate_freqs_iterative_fft(
            signal, self.n_sinusoids, self.fs, self.n_fft_iip
        )
        if rough_freqs.size == 0:
            warnings.warn("Initial frequency estimation (IIp-DFT) failed.")
            return np.array([])

        if np.isrealobj(signal):
            # For real signals, consider positive and negative frequency
            # pairs
            kernel_freqs = np.concatenate([rough_freqs, -rough_freqs])
        else:
            # For complex signals, use only estimated frequencies
            kernel_freqs = -rough_freqs

        # 2. Build a truncated DFT kernel from the coarse frequency
        #    estimate.  (Corresponds to Alg. 4, Step 2)
        n_snapshots = signal.size - self.subspace_dim + 1
        kernel_matrix = self._build_vandermonde_matrix(
            kernel_freqs, n_snapshots, self.fs
        )

        # 3. Project data onto the kernel via fast convolution to get
        #    Yp. (Corresponds to Alg. 4, Step 3, implemented with
        #    Alg. 3)
        projected_matrix = self._fast_hankel_vandermonde_product(signal, kernel_matrix)

        # 4. Orthonormalize the approximated signal subspace via QR
        #    decomposition. (Corresponds to Alg. 4, Step 4)
        try:
            # Performs efficient thin QR decomposition
            # with "economic" mode
            _q_matrix, _ = qr(projected_matrix, mode="economic")
            q_matrix = _q_matrix.astype(np.complex128)
        except np.linalg.LinAlgError:
            warnings.warn("QR decomposition failed in FFT-ESPRIT.")
            return np.array([])

        # 5. Apply a standard ESPRIT solver to the approximated subspace
        #    Q. (Corresponds to Alg. 4, Steps 5-9)
        omegas = self.solver.solve(q_matrix)

        # 6. Post-process the results to get final frequencies in Hz.
        est_freqs = self._postprocess_omegas(omegas)
        return est_freqs

    def _fast_hankel_vandermonde_product(
        self,
        signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        kernel_matrix: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        """Compute the product of a Hankel matrix and a kernel matrix.

        This method efficiently calculates `Yp = X @ Ap` where `X` is
        the Hankel matrix of the signal. It leverages the convolution
        theorem, replacing the direct, computationally expensive matrix
        multiplication with FFT-based convolution via
        `scipy.signal.fftconvolve`.

        This corresponds to the "Fast Hankel Matrix-Matrix product"
        (Algorithm 3) in the reference paper.

        Args:
            signal (np.ndarray):
                The input signal x, of length N.
            kernel_matrix (np.ndarray):
                The Vandermonde-like kernel matrix Ap, of shape (L, P).

        Returns:
            np.ndarray:
                The projected matrix Yp = X @ Ap, of shape (M, P).
        """
        n_components = kernel_matrix.shape[1]
        projected_matrix = np.zeros(
            (self.subspace_dim, n_components), dtype=np.complex128
        )
        for i in range(n_components):
            kernel_vector = kernel_matrix[:, i]
            conv_result = fftconvolve(signal, kernel_vector[::-1], mode="valid")
            projected_matrix[:, i] = conv_result[: self.subspace_dim]
        return projected_matrix

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include the name of the
        solver class and the length of iterative interpolation FFT.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["solver"] = self.solver.__class__.__name__
        params["n_fft_iip"] = self.n_fft_iip
        return params
