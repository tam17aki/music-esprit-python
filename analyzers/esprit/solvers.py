# -*- coding: utf-8 -*-
"""Defines solver classes for Starndard ESPRIT and Unitary ESPRIT.

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

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError, eigvals, pinv, svd
from scipy.sparse import csc_array, csr_array


# pylint: disable=too-few-public-methods
class LSEspritSolver:
    """A solver class for the ESPRIT rotational operator using the Least Squares."""

    def solve(
        self, signal_subspace: npt.NDArray[np.complex128] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Solves for rotational factors using the Least Squares (LS) method.

        This method estimates the rotation operator Psi from the equation
        `subspace_upper @ Psi = subspace_lower` by solving a least squares
        problem. The normalized angular frequencies are then computed from
        the phase angles of the eigenvalues of Psi.

        Args:
            signal_subspace (np.ndarray):
                The complex-valued signal subspace `Es`. Shape: (L, 2M).

        Returns:
            np.ndarray:
                An array of estimated normalized angular frequencies (omegas)
                in radians per sample. Shape: (2M,).
                Returns an empty array if estimation fails.
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
            warnings.warn(
                "Eigenvalue decomposition failed while solving rotation operator."
            )
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        return np.angle(eigenvalues).astype(np.float64)


# pylint: disable=too-few-public-methods
class TLSEspritSolver:
    """A solver class for the ESPRIT rotational operator using the Total LS."""

    def solve(
        self, signal_subspace: npt.NDArray[np.complex128] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Solves for rotational factors using the Total Least Squares (TLS) method.

        This method formulates the problem as `[subspace_upper, subspace_lower]`
        and solves for the rotational operator Psi via Singular Value
        Decomposition (SVD). This approach is generally more robust in noisy
        conditions than the LS method. The normalized angular frequencies are
        computed from the phase angles of the eigenvalues of Psi.

        Args:
            signal_subspace (np.ndarray):
                The signal subspace `Es` (complex128 or float64). Shape: (L, 2M).

        Returns:
            np.ndarray:
                An array of estimated normalized angular frequencies (omegas)
                in radians per sample (complex128 or float64). Shape: (2M,).
                Returns an empty array if estimation fails.
        """
        # Form the augmented matrix for SVD
        subspace_upper = signal_subspace[:-1, :]
        subspace_lower = signal_subspace[1:, :]
        _augmented_subspace = np.concatenate((subspace_upper, subspace_lower), axis=1)
        if np.isdtype(np.float64, signal_subspace.dtype):
            augmented_subspace = _augmented_subspace.astype(np.float64)
        else:
            augmented_subspace = _augmented_subspace.astype(np.complex128)

        try:
            _, _, vh = svd(augmented_subspace)
        except LinAlgError:
            warnings.warn("SVD on augmented_subspace did not converge.")
            return np.array([])

        # Partition the Vh matrix to solve the TLS problem
        model_order = subspace_upper.shape[1]
        v12 = vh[model_order:, :model_order]
        v22 = vh[model_order:, model_order:]

        # Solve the rotation operator
        try:
            rotation_operator = -v12 @ pinv(v22)
        except LinAlgError:
            warnings.warn(
                "TLS matrix inversion failed while computing rotation operator."
            )
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn(
                "Eigenvalue decomposition failed while solving rotation operator."
            )
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        return np.angle(eigenvalues).astype(np.float64)


class _UnitaryEspritHelpers:  # pylint: disable=too-few-public-methods
    """A Mixin class to provide helper methods for Unitary ESPRIT solvers."""

    @staticmethod
    def _get_unitary_transform_matrix(subspace_dim: int) -> npt.NDArray[np.complex128]:
        """Construct the unitary matrix Q for real-valued transform.

        Args:
           subspace_dim (int): Dimension of signal subspace.

        Returns:
           np.ndarray: Unitary matrix for the tranform (complex128).
        """
        p = subspace_dim // 2
        identity = np.eye(p, dtype=np.complex128)
        exchange = identity[:, ::-1]  # exchange matrix
        if subspace_dim % 2 == 0:  # L is even
            q_matrix = np.block([[identity, 1j * identity], [exchange, -1j * exchange]])
        else:  #  L is odd
            q_left = np.vstack(
                [identity, np.zeros((1, p)), exchange], dtype=np.complex128
            )
            q_center = np.zeros((subspace_dim, 1), dtype=np.complex128)
            q_center[p] = np.sqrt(2.0).astype(np.complex128)
            q_right = np.vstack(
                [1j * identity, np.zeros((1, p)), -1j * exchange], dtype=np.complex128
            )
            q_matrix = np.hstack([q_left, q_center, q_right])
        return q_matrix

    def _get_real_selection_matrices(
        self, subspace_dim: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Construct the real-valued selection matrices K1 and K2.

        Args:
           subspace_dim (int): Dimension of signal subspace.

        Returns:
           tuple[np.ndarray, np.ndarray]: The selection matrices K1 and K2.
        """
        m_prime = subspace_dim - 1  # Subarray size

        # 1. Build a complex selection matrix J1 (select the first m' rows)
        j1 = np.eye(subspace_dim, dtype=np.complex128)[:m_prime, :]

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
    """A solver class for the real-valued Unitary ESPRIT problem using least squares.

    This solver takes a real-valued signal subspace and solves a generalized
    eigenvalue problem to find the frequencies based on least squares approach.
    """

    def solve(
        self, signal_subspace: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Solves the real-valued Unitary ESPRIT problem using Least Squares.

        This method constructs real-valued selection matrices K1 and K2,
        and solves the system `(K1 @ Es) @ Y = (K2 @ Es)` for Y using a
        least squares approach. The normalized angular frequencies (omegas) are
        recovered from the eigenvalues of Y using the arctangent function.

        Args:
            signal_subspace (np.ndarray):
                The real-valued signal subspace `Es_real` (float64). Shape: (L, 2M).

        Returns:
            np.ndarray:
                An array of estimated normalized angular frequencies (omegas)
                in radians per sample (float64). Shape: (M,).
                Returns an empty array if estimation fails.
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
            warnings.warn("Eigenvalue decomposition of Y_LS failed.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas = 2 * np.arctan(np.real(eigenvalues))
        return omegas.astype(np.float64)


# pylint: disable=too-few-public-methods
class TLSUnitaryEspritSolver(_UnitaryEspritHelpers):
    """A solver class for the real-valued ESPRIT problem using total least squares.

    This solver takes a real-valued signal subspace and solves a generalized
    eigenvalue problem to find the frequencies based on total least squares approach.
    """

    def solve(
        self, signal_subspace: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Solves the real-valued Unitary ESPRIT problem using Total Least Squares.

        This method constructs real-valued matrices T1 and T2 from the real
        signal subspace, and solves the system `T1 @ Y ≈ T2` using a more
        robust Total Least Squares approach via SVD. The normalized angular
        frequencies (omegas) are recovered from the eigenvalues of the resulting
        solution matrix Y_TLS using the arctangent function.

        Args:
            signal_subspace (np.ndarray):
                The real-valued signal subspace `Es_real` (float64). Shape: (L, 2M).

        Returns:
            np.ndarray:
                An array of estimated normalized angular frequencies (omegas)
                in radians per sample (float64). Shape: (M,).
                Returns an empty array if estimation fails.
        """
        subspace_dim = signal_subspace.shape[0]
        k1, k2 = self._get_real_selection_matrices(subspace_dim)
        t1 = k1 @ signal_subspace
        t2 = k2 @ signal_subspace

        try:
            _, _, vh = svd(np.concatenate((t1, t2), axis=1))
        except LinAlgError:
            warnings.warn("SVD on augmented_subspace did not converge.")
            return np.array([])

        model_order = t1.shape[1]
        v12 = vh[model_order:, :model_order]
        v22 = vh[model_order:, model_order:]

        # Solve the rotation operator
        try:
            rotation_operator = -v12 @ pinv(v22)
        except LinAlgError:
            warnings.warn(
                "TLS matrix inversion failed while computing rotation operator."
            )
            return np.array([])
        try:
            eigenvalues = eigvals(rotation_operator)
        except LinAlgError:
            warnings.warn("Eigenvalue decomposition of Y_TLS failed.")
            return np.array([])

        # Recover normalized angular frequencies from eigenvalues
        omegas = 2 * np.arctan(np.real(eigenvalues))
        return omegas.astype(np.float64)
