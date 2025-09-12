# -*- coding: utf-8 -*-
"""Defines AnalyzerBase class, the abstract base class for analyzers.

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
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError, hankel, pinv

from utils.data_models import SinusoidParameters

RATIO_UPPER = 0.5


class AnalyzerBase(ABC):
    """Abstract base class for parameter analyzers."""

    def __init__(self, fs: float, n_sinusoids: int, subspace_ratio: float = 1 / 3):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            subspace_ratio (float, optional): The ratio of the subspace dimension
                to the signal length. Should be between 0 and 0.5. Defaults to 1/3.
        """
        if not 0 < subspace_ratio <= RATIO_UPPER:
            raise ValueError(f"subspace_ratio must be in the range (0, {RATIO_UPPER}].")

        self.fs: float = fs
        self.n_sinusoids: int = n_sinusoids
        self.subspace_ratio: float = subspace_ratio
        self.subspace_dim: int = -1
        self.est_params: SinusoidParameters | None = None

    def fit(self, signal: npt.NDArray[np.complex128]) -> Self:
        """Run the full parameter estimation process.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            Self@AnalyzerBase: The fitted object.
                Returns empty result if estimation fails.
        """
        n_samples = signal.size
        model_order = 2 * self.n_sinusoids
        self.subspace_dim = int(n_samples * self.subspace_ratio)
        if (
            self.subspace_dim <= model_order
            or self.subspace_dim >= n_samples - model_order
        ):
            self.est_params = SinusoidParameters(
                np.array([]), np.array([]), np.array([])
            )
            warnings.warn("Invalid subspace dimension. Returning empty result.")
            return self
        freqs = self._estimate_frequencies(signal)
        if freqs.size != self.n_sinusoids:
            self.est_params = SinusoidParameters(freqs, np.array([]), np.array([]))
            warnings.warn(
                f"Expected {self.n_sinusoids} components, but found {freqs.size}."
            )
            return self
        amps, phases = self._estimate_amplitudes_phases(signal, freqs)
        self.est_params = SinusoidParameters(freqs, amps, phases)
        return self

    @abstractmethod
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies using a specific MUSIC variant."""
        raise NotImplementedError

    @staticmethod
    def _build_covariance_matrix(
        signal: npt.NDArray[np.complex128], subspace_dim: int
    ) -> npt.NDArray[np.complex128]:
        """Build the covariance matrix from the input signal.

        Args:
            signal (np.ndarray): Input signal (complex128).
            subspace_dim (int): The dimension of subspace.

        Returns:
            np.ndarray: The covariance matrix (complex128).
        """
        n_samples = signal.size
        n_snapshots = n_samples - subspace_dim + 1
        hankel_matrix = hankel(signal[:subspace_dim], signal[subspace_dim - 1 :])
        cov_matrix = (hankel_matrix @ hankel_matrix.conj().T) / n_snapshots
        return cov_matrix

    def _estimate_amplitudes_phases(
        self,
        signal: npt.NDArray[np.complex128],
        estimated_freqs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Estimate amplitudes and phases from frequencies using least squares.

        Args:
            signal (np.ndarray): Input signal (complex128).
            estimated_freqs (np.ndarray): Array of estimated frequencies in Hz.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - estimated_amps (np.ndarray): Estimated amplitudes.
                - estimated_phases (np.ndarray): Estimated phases in radians.
        """
        n_samples = signal.size
        n_sinusoids = estimated_freqs.size

        # 1. Build the Vandermonde matrix V
        t = np.arange(n_samples) / self.fs
        vandermonde_matrix = np.zeros((n_samples, n_sinusoids), dtype=np.complex128)
        for i, freq in enumerate(estimated_freqs):
            vandermonde_matrix[:, i] = np.exp(2j * np.pi * freq * t)

        # 2. Solve for complex amplitudes c using pseudo-inverse
        # y = V @ c  =>  c = pinv(V) @ y
        try:
            complex_amps = pinv(vandermonde_matrix) @ signal
        except LinAlgError:
            warnings.warn("Least squares estimation for amplitudes/phases failed.")
            return np.array([]), np.array([])

        # 3. Extract amplitudes and phases
        # For a real-valued sinusoid A*cos(2*pi*f*t + phi), the complex amplitude
        # estimated using only the positive frequency is (A/2)*exp(j*phi).
        # Therefore, we need to multiply the magnitude by 2.
        estimated_amps = 2 * np.abs(complex_amps).astype(np.float64)
        estimated_phases = np.angle(complex_amps).astype(np.float64)

        # Sort results according to frequency for consistent comparison
        sort_indices = np.argsort(estimated_freqs)

        return estimated_amps[sort_indices], estimated_phases[sort_indices]

    @property
    def frequencies(self) -> npt.NDArray[np.float64]:
        """Return the estimated frequencies in Hz after fitting."""
        if self.est_params is None:
            raise AttributeError("Cannot access 'frequencies' before running fit().")
        return self.est_params.frequencies

    @property
    def amplitudes(self) -> npt.NDArray[np.float64]:
        """Return the estimated amplitudes after fitting."""
        if self.est_params is None or self.est_params.amplitudes.size == 0:
            raise AttributeError(
                "Cannot access 'amplitudes' before fitting is complete."
            )
        return self.est_params.amplitudes

    @property
    def phases(self) -> npt.NDArray[np.float64]:
        """Return the estimated phases in radians after fitting."""
        if self.est_params is None or self.est_params.phases.size == 0:
            raise AttributeError("Cannot access 'phases' before fitting is complete.")
        return self.est_params.phases
