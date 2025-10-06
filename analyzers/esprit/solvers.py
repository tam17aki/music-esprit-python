# -*- coding: utf-8 -*-
"""Defines solver classes for ESPRIT variants.

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
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from scipy.linalg import LinAlgError, eigvals, pinv, svd
from scipy.sparse import csc_array, csr_array

from utils.data_models import (
    ComplexArray,
    FloatArray,
    NumpyComplex,
    NumpyFloat,
)

from .._common import ZERO_LEVEL

EspritSolverType: TypeAlias = Literal["ls", "tls"]
FastEspritSolverType: TypeAlias = Literal["ls", "tls", "woodbury"]


# pylint: disable=too-few-public-methods
@runtime_checkable
class EspritSolver(Protocol):
    """Protocol for standard (complex or real) ESPRIT solvers."""

    def solve(self, signal_subspace: FloatArray | ComplexArray) -> FloatArray:
        """Estimate angular frequencies from a signal subspace."""
        ...  # pylint: disable=unnecessary-ellipsis


# pylint: disable=too-few-public-methods
@runtime_checkable
class UnitaryEspritSolver(Protocol):
    """Protocol for Unitary (strictly real-valued) ESPRIT solvers."""

    def solve(self, signal_subspace: FloatArray) -> FloatArray:
        """Estimate angular frequencies from a signal subspace."""
        ...  # pylint: disable=unnecessary-ellipsis


# pylint: disable=too-few-public-methods
@runtime_checkable
class FastEspritSolver(Protocol):
    """Protocol for solvers compatible with fast ESPRIT variants."""

    def solve(self, signal_subspace: ComplexArray) -> FloatArray:
        """Estimate angular frequencies from a signal subspace."""
        ...  # pylint: disable=unnecessary-ellipsis


# pylint: disable=too-few-public-methods
class LSEspritSolver:
    """Solves the ESPRIT rotational invariance equation using LS."""

    def solve(self, signal_subspace: FloatArray | ComplexArray) -> FloatArray:
        """Estimate angular frequencies from a signal subspace via LS.

        This method solves for the rotational operator Psi in the
        equation `subspace_upper @ Psi = subspace_lower` using a
        least-squares fit. Angular frequencies (omegas) are then derived
        from the phase angles of Psi's eigenvalues.

        Args:
            signal_subspace (FloatArray | ComplexArray):
                The signal subspace `Es`. Shape: (L, 2M).

        Returns:
            FloatArray:
                Estimated normalized angular frequencies in rad/sample.
                Shape: (2M,).
                Returns an empty array on failure.
        """
        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]

        # Solve the rotation operator
        try:
            rotation_operator = pinv(subspace_upper) @ subspace_lower
        except LinAlgError:
            warnings.warn("Matrix inversion failed in parameter solving.")
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("EVD failed while solving rotation operator.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas = np.angle(eigenvalues).astype(NumpyFloat)
        return omegas


# pylint: disable=too-few-public-methods
class TLSEspritSolver:
    """Solves the ESPRIT rotational invariance equation using TLS."""

    def solve(self, signal_subspace: FloatArray | ComplexArray) -> FloatArray:
        """Estimate angular frequencies from a signal subspace via TLS.

        This method solves for the rotational operator Psi by
        formulating the problem for a Total Least Squares fit,
        typically solved via SVD. This approach can be more robust
        to noise than standard LS. Frequencies are derived from the
        eigenvalues of Psi.

        Args:
            signal_subspace (FloatArray | ComplexArray):
                The signal subspace `Es`. Shape: (L, 2M).

        Returns:
            FloatArray:
                Estimated normalized angular frequencies in rad/sample.
                Shape: (2M,).
                Returns an empty array on failure.
        """
        # Form the augmented matrix for SVD
        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]
        _augmented_subspace = np.concatenate(
            (subspace_upper, subspace_lower), axis=1
        )
        if np.isrealobj(signal_subspace):
            augmented_subspace = _augmented_subspace.astype(NumpyFloat)
        else:
            augmented_subspace = _augmented_subspace.astype(NumpyComplex)

        try:
            _, _, vh = svd(augmented_subspace, full_matrices=False)
        except LinAlgError:
            warnings.warn("SVD on augmented_subspace did not converge.")
            return np.array([])

        # Partition the Vh matrix to solve the TLS problem
        model_order = subspace_upper.shape[1]
        v11 = vh[:model_order, :model_order]
        v12 = vh[:model_order, model_order:]

        # Solve the rotation operator
        try:
            rotation_operator = pinv(v11) @ v12
        except LinAlgError:
            warnings.warn(
                "TLS matrix inversion failed while computing rotation "
                + "operator."
            )
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("EVD failed while solving rotation operator.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas = np.angle(eigenvalues).astype(NumpyFloat)
        return omegas


# pylint: disable=too-few-public-methods
class _UnitaryEspritHelpers:
    """Mixin class providing helpers for Unitary ESPRIT solvers."""

    @staticmethod
    def _get_unitary_transform_matrix(subspace_dim: int) -> ComplexArray:
        """Construct the unitary matrix Q for real-valued transform.

        Args:
            subspace_dim (int): Dimension of signal subspace.

        Returns:
            ComplexArray: Unitary matrix for the tranform.
        """
        p = subspace_dim // 2
        identity = np.eye(p, dtype=NumpyComplex)
        exchange = identity[:, ::-1]  # exchange matrix
        if subspace_dim % 2 == 0:  # L is even
            q_matrix = np.block(
                [[identity, 1j * identity], [exchange, -1j * exchange]]
            )
        else:  #  L is odd
            q_left = np.vstack(
                [identity, np.zeros((1, p)), exchange], dtype=NumpyComplex
            )
            q_center = np.zeros((subspace_dim, 1), dtype=NumpyComplex)
            q_center[p] = np.sqrt(2.0).astype(NumpyComplex)
            q_right = np.vstack(
                [1j * identity, np.zeros((1, p)), -1j * exchange],
                dtype=NumpyComplex,
            )
            q_matrix = np.hstack([q_left, q_center, q_right])
        return q_matrix

    def _get_real_selection_matrices(
        self, subspace_dim: int
    ) -> tuple[FloatArray, FloatArray]:
        """Construct the real-valued selection matrices K1 and K2.

        Args:
            subspace_dim (int): Dimension of signal subspace.

        Returns:
            tuple[FloatArray, FloatArray]:
                The selection matrices K1 and K2.
        """
        m_prime = subspace_dim - 1  # Subarray size

        # 1. Build a complex selection matrix J1
        #    (select the first m' rows)
        j1 = np.eye(subspace_dim, dtype=NumpyComplex)[:m_prime, :]

        # 2. Build unitary transformation matrix Q
        q_m_prime = csc_array(self._get_unitary_transform_matrix(m_prime))
        q_m = csr_array(self._get_unitary_transform_matrix(subspace_dim))

        # 3. Calculate K1 and K2 according to equation (32).
        # K1 = Q_m'^H * (J1 + Π_m' * J1 * Π_M) * Q_M
        temp_array = csr_array(np.fliplr(np.flipud(j1)))  # Π_m' * J1 * Π_M
        j1_array = csr_array(j1)
        k1_term = j1_array + temp_array
        k1 = q_m_prime.conj().T @ k1_term @ q_m

        # K2 = Q_m'^H * j * (J1 - Π_m' * J1 * Π_M) * Q_M
        k2_term = 1j * (j1_array - temp_array)
        k2 = q_m_prime.conj().T @ k2_term @ q_m

        return np.real(k1.toarray()), np.real(k2.toarray())


# pylint: disable=too-few-public-methods
class LSUnitaryEspritSolver(_UnitaryEspritHelpers):
    """Solves the real-valued Unitary ESPRIT problem via LS.

    This solver takes a real-valued signal subspace and solves a
    generalized eigenvalue problem to find the frequencies based on
    least squares approach.
    """

    def solve(self, signal_subspace: FloatArray) -> FloatArray:
        """Estimate omegas from real subspace via Unitary ESPRIT (LS).

        This method constructs real-valued selection matrices K1 and K2,
        and solves the system `(K1 @ Es) @ Y = (K2 @ Es)` for Y using a
        least squares approach. The normalized angular frequencies
        (omegas) are recovered from the eigenvalues of Y using the
        arctangent function.

        Args:
            signal_subspace (FloatArray):
                The real-valued signal subspace `Es_real`.
                Shape: (L, 2M).

        Returns:
            FloatArray:
                Estimated normalized angular frequencies in rad/sample.
                Shape: (M,).
                Returns an empty array on failure.
        """
        subspace_dim = signal_subspace.shape[0]
        k1, k2 = self._get_real_selection_matrices(subspace_dim)
        t1 = k1 @ signal_subspace
        t2 = k2 @ signal_subspace

        # Solve the rotation operator
        try:
            rotation_operator = pinv(t1) @ t2
        except LinAlgError:
            warnings.warn("Least Squares problem in Unitary ESPRIT failed.")
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("EVD of Y_LS failed.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas: FloatArray = 2 * np.arctan(np.real(eigenvalues))
        return omegas


# pylint: disable=too-few-public-methods
class TLSUnitaryEspritSolver(_UnitaryEspritHelpers):
    """Solves the real-valued Unitary ESPRIT problem via TLS.

    This solver takes a real-valued signal subspace and solves a
    generalized eigenvalue problem to find the frequencies based on
    total least squares approach.
    """

    def solve(self, signal_subspace: FloatArray) -> FloatArray:
        """Estimate omegas from real subspace via Unitary ESPRIT (TLS).

        This method constructs real-valued matrices T1 and T2 from the
        subspace and solves the system `T1 @ Y ≈ T2` using a robust
        Total Least Squares approach via SVD. Normalized angular
        frequencies are recovered from the eigenvalues of the solution
        matrix Y_TLS using the arctangent function.

        Args:
            signal_subspace (FloatArray):
                The real-valued signal subspace `Es_real`.
                Shape: (L, 2M).

        Returns:
            FloatArray:
                Estimated normalized angular frequencies in rad/sample.
                Shape: (M,).
                Returns an empty array on failure.
        """
        subspace_dim = signal_subspace.shape[0]
        k1, k2 = self._get_real_selection_matrices(subspace_dim)
        t1 = k1 @ signal_subspace
        t2 = k2 @ signal_subspace

        try:
            _, _, vh = svd(
                np.concatenate((t1, t2), axis=1), full_matrices=False
            )
        except LinAlgError:
            warnings.warn("SVD on augmented_subspace did not converge.")
            return np.array([])

        model_order = t1.shape[1]
        v11 = vh[:model_order, :model_order]
        v12 = vh[:model_order, model_order:]

        # Solve the rotation operator
        try:
            rotation_operator = pinv(v11) @ v12
        except LinAlgError:
            warnings.warn(
                "TLS matrix inversion failed while computing rotation "
                + "operator."
            )
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("EVD of Y_TLS failed.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas: FloatArray = 2 * np.arctan(np.real(eigenvalues))
        return omegas


# pylint: disable=too-few-public-methods
class WoodburyLSEspritSolver:
    """Implements a fast LS-ESPRIT solver using the Woodbury formula.

    This solver is optimized for orthonormal signal subspaces (Q), such
    as those produced by FFT-ESPRIT's QR decomposition. It avoids a
    direct pseudo-inverse, offering higher computational
    efficiency. Corresponds to the solver described in Algorithm 4 in
    Kiser et al. (2023).
    """

    def solve(self, signal_subspace: ComplexArray) -> FloatArray:
        """Solve the LS problem for an orthonormal subspace efficiently.

        This method implements a fast version of the LS solver by
        applying the Sherman-Morrison formula (a rank-1 update) to avoid
        a direct pseudo-inverse calculation. It is tailored for cases
        where the input signal subspace is orthonormal.

        Args:
            signal_subspace (ComplexArray):
                The orthonormal signal subspace matrix Q.
                Shape: (L, 2M).

        Returns:
            FloatArray:
                Estimated normalized angular frequencies in rad/sample.
                Shape: (M,).
                Returns an empty array on failure.
        """
        q_matrix = signal_subspace
        q_upper = q_matrix[:-1, :]
        q_lower = q_matrix[1:, :]

        # 1. Extract q (the last row vector)
        q_last_row = q_matrix[-1, :].reshape(1, -1)  # (1, 2M)

        # 2. Calculate q*q^H (scalar)
        #    Since q_matrix is orthonormal, ||q_last_row||^2 <= 1
        q_q_h = np.dot(
            q_last_row, q_last_row.conj().T
        ).item()  # .item() to scalar

        # 3. Calculate the coefficients needed to calculate
        #    (I - q^H*q)^-1. Coefficient = 1 / (1 - q*q^H)
        denominator = 1 - q_q_h
        if np.abs(denominator) < ZERO_LEVEL:
            warnings.warn("Denominator in Woodbury formula is close to zero.")
            # In this case, (I - q^H*q) is a nearly singular matrix.
            # It is safe to fall back to the LS solution using pinv.
            rotation_operator = pinv(q_upper) @ q_lower
        else:
            # 4. Calculate (Q↑^H*Q↑)^-1
            #    inv_matrix = I + q^H*q / (1 - q*q^H)
            q_h_q = q_last_row.conj().T @ q_last_row
            inv_matrix = np.eye(q_matrix.shape[1]) + q_h_q / denominator

            # 5. Calculate Q↑^H*Q↓
            q_upper_h_q_lower = q_upper.conj().T @ q_lower

            # 6. Calculate the final rotation operator Ψ
            rotation_operator = inv_matrix @ q_upper_h_q_lower

        # Calculates eigenvalues and returns frequencies
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("EVD failed in Woodbury solver.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas = np.angle(eigenvalues).astype(NumpyFloat)
        return omegas
