# -*- coding: utf-8 -*-
"""Defines NystromEspritAnalyzer class for the Nyström-based ESPRIT method.

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
from typing import final, get_args, override

import numpy as np
from scipy.linalg import LinAlgError, eigh, qr

from utils.data_models import (
    ComplexArray,
    FloatArray,
    NumpyComplex,
    NumpyFloat,
    SignalArray,
)

from .._common import ZERO_LEVEL
from ..models import AnalyzerParameters
from .base import EVDBasedEspritAnalyzer
from .solvers import EspritSolverType, esprit_solvers


@final
class NystromEspritAnalyzer(EVDBasedEspritAnalyzer):
    """Implements the Nyström-based ESPRIT algorithm.

    This analyzer uses the Nyström method to efficiently approximate
    the signal subspace from a smaller, sampled portion of the
    covariance matrix. This avoids the expensive EVD of the full
    matrix, making it suitable for large datasets.

    Reference:
        C. Qian, et al., "Computationally efficient ESPRIT algorithm...
        based on Nyström method," Signal Processing, 2014.
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        solver: EspritSolverType = "ls",
        *,
        nystrom_rank_factor: int = 10,
    ) -> None:
        """Initialize the Nyström-ESPRIT analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            solver (EspritSolverType, optional): The numerical solver to
                use for the core rotational invariance equation. Can be
                "ls" (Least Squares) or "tls" (Total Least Squares).
                Defaults to "ls".
            nystrom_rank_factor (int, optional): A factor to determine
                the number of rows to sample for the Nyström
                approximation (K = factor * 2M). A larger value improves
                robustness at the cost of computation. Defaults to 10.
        """
        super().__init__(fs, n_sinusoids)
        valid_solvers = get_args(EspritSolverType)
        if solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver '{solver}'. Choose from {valid_solvers}."
            )
        self.solver = esprit_solvers[solver]
        self.solver_name = solver
        self.nystrom_rank_factor = nystrom_rank_factor

    @override
    def _estimate_frequencies(self, signal: SignalArray) -> FloatArray:
        """Estimate signal frequencies using the Nyström-ESPRIT method.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            FloatArray: An array of estimated frequencies in Hz.
                Returns an empty array on failure.
        """
        # 1. Estimate the signal subspace
        signal_subspace = self._estimate_signal_subspace(signal)
        if signal_subspace is None:
            return np.array([])

        # 2. Solve frequencies with the stored solver
        omegas = self.solver(signal_subspace)

        # 3. Post-processes raw angular frequencies to final frequency
        #    estimates
        est_freqs = self._postprocess_omegas(omegas)

        return est_freqs

    @override
    def _estimate_signal_subspace(
        self, signal: SignalArray
    ) -> FloatArray | ComplexArray | None:
        """Approximate the signal subspace using the Nyström method.

        This method avoids a full covariance EVD by approximating the
        subspace from smaller, sampled sub-matrices. This significantly
        reduces computational complexity compared to standard ESPRIT.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            FloatArray | ComplexArray | None: An orthonormal basis for
                the approximated signal subspace. Returns None on
                failure.
        """
        # --- Step 1: Prepare the parameters ---
        # The number of complex exponential components
        if np.isrealobj(signal):
            # For real signals, positive and negative frequency pairs
            # are considered
            n_complex_sinusoids = 2 * self.n_sinusoids
        else:
            # For complex signals, the number of signals themselves
            n_complex_sinusoids = self.n_sinusoids

        # The rank of the Nyström method
        k_nystrom = self.nystrom_rank_factor * n_complex_sinusoids
        if not n_complex_sinusoids < k_nystrom < self.subspace_dim:
            warnings.warn(
                "Invalid model order for Nyström method. "
                + "Adjust nystrom_rank_factor."
            )
            k_nystrom = int((n_complex_sinusoids + self.subspace_dim) / 2)

        # Calculate the data matrix
        data_matrix = self._build_hankel_matrix(signal, self.subspace_dim)

        # --- Step 2: Calculate R11 and R21 ---
        r11, r21 = self._compute_sub_covariance_matrices(
            data_matrix, k_nystrom
        )

        # --- Step 3: Build the matrix G ---
        try:
            matrix_g = self._build_g_matrix(r11, r21)
        except LinAlgError:
            warnings.warn("Failed to build the G matrix in Nyström method.")
            return None

        # --- Step 4: Compute the signal subspace from G ---
        try:
            signal_subspace = self._compute_subspace_from_g(
                matrix_g, n_complex_sinusoids
            )
        except LinAlgError:
            warnings.warn(
                "A linear algebra error occurred during the Nyström subspace "
                + "computation (e.g., EVD or matrix inversion). "
                + "The signal may be too noisy or the parameters inadequate."
            )
            return None

        return signal_subspace

    @staticmethod
    def _compute_sub_covariance_matrices(
        data_matrix: FloatArray | ComplexArray, model_order: int
    ) -> tuple[FloatArray | ComplexArray, FloatArray | ComplexArray]:
        """Compute the R11 and R21 sub-matrices.

        This function partitions the data matrix into X1 (first
        `model_order` rows) and X2 (remaining rows) and computes the
        sample covariance matrices R11 = E[X1*X1^H] and
        R21 = E[X2*X1^H].

        Args:
            data_matrix (FloatArray | ComplexArray):
                The full Hankel data matrix X.
            model_order (int):
                The number of rows to sample for the approximation (K).

        Returns:
            tuple[FloatArray | ComplexArray, FloatArray | ComplexArray]:
                A tuple containing the (R11, R21) matrices.
        """
        x1 = data_matrix[:model_order, :]
        x2 = data_matrix[model_order:, :]
        n_snapshots = data_matrix.shape[1]
        r11 = (x1 @ x1.conj().T) / n_snapshots
        r21 = (x2 @ x1.conj().T) / n_snapshots
        if np.isrealobj(r11):
            r11_float: FloatArray = r11.astype(NumpyFloat)
            r21_float: FloatArray = r21.astype(NumpyFloat)
            return r11_float, r21_float
        r11_complex: ComplexArray = r11.astype(NumpyComplex)
        r21_complex: ComplexArray = r21.astype(NumpyComplex)
        return r11_complex, r21_complex

    @staticmethod
    def _build_g_matrix(
        r11: FloatArray | ComplexArray, r21: FloatArray | ComplexArray
    ) -> FloatArray | ComplexArray:
        """Build the intermediate matrix G based on the Nyström method.

        This corresponds to G = U @ Lambda^{1/2} in the proposition 1 in
        the reference paper. The matrix square root inverse is
        calculated via a stable eigenvalue decomposition of R11.

        Args:
            r11 (FloatArray | ComplexArray):
                The K x K sub-covariance matrix.
            r21 (FloatArray | ComplexArray):
                The (L-K) x K sub-covariance matrix.

        Returns:
            FloatArray | ComplexArray:
                The resulting L x K matrix G.
        """
        eigvals_r11, u11 = eigh(r11)
        idx = np.argsort(eigvals_r11)[::-1]
        eigvals_r11 = eigvals_r11[idx]
        u11 = u11[:, idx]
        safe_eigvals = np.maximum(eigvals_r11, ZERO_LEVEL)
        lambda11_inv = np.diag(1 / safe_eigvals)
        u21 = r21 @ u11 @ lambda11_inv
        u = np.vstack([u11, u21])
        _g_matrix = u @ np.diag(np.sqrt(safe_eigvals))
        if np.isrealobj(_g_matrix):
            g_matrix_float: FloatArray = _g_matrix.astype(NumpyFloat)
            return g_matrix_float
        g_matrix_complex: ComplexArray = _g_matrix.astype(NumpyComplex)
        return g_matrix_complex

    @staticmethod
    def _compute_subspace_from_g(
        matrix_g: FloatArray | ComplexArray, n_components: int
    ) -> FloatArray | ComplexArray:
        """Compute the signal subspace from the G matrix.

        This function implements Proposition 1 from the reference paper.
        It performs an EVD of G^H*G and computes the final subspace Pi =
        G * U_G, which is then orthonormalized via QR decomposition.

        Args:
            matrix_g (FloatArray | ComplexArray):
                The intermediate matrix G of shape (L, K).

            n_components (int):
                Number of complex exponential components to estimate.
                This is `2 * n_sinusoids` for real signals (to account
                for +/- frequency pairs) and `n_sinusoids` for complex
                signals.

        Returns:
            FloatArray | ComplexArray:
                An orthonormal basis for the approximated signal
                subspace, Q.
        """
        g_h_g = matrix_g.conj().T @ matrix_g
        eigvals_g, u_g = eigh(g_h_g)  # eigenvalues Lambda_G, basis U_G
        idx = np.argsort(eigvals_g)[::-1]
        u_g = u_g[:, idx]
        signal_subspace_unortho = (matrix_g @ u_g)[:, :n_components]
        _q_matrix, _ = qr(signal_subspace_unortho, mode="economic")
        if np.isrealobj(_q_matrix):
            q_matrix_float: FloatArray = _q_matrix.astype(NumpyFloat)
            return q_matrix_float
        q_matrix_complex: ComplexArray = _q_matrix.astype(NumpyComplex)
        return q_matrix_complex

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include the name of the
        solver class and the hyperparameter specific to the
        Nyström-based ESPRIT method.

        Returns:
            AnalyzerParameters:
                A TypedDict containing both common and method-specific
                hyperparameters.
        """
        params = super().get_params()
        params["solver"] = self.solver_name
        params["nystrom_rank_factor"] = self.nystrom_rank_factor
        return params
