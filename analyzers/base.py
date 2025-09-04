# -*- coding: utf-8 -*-
"""The abstract base class for analyzers.

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
from scipy.linalg import hankel, pinv

from utils.data_models import SinusoidParameters


class AnalyzerBase(ABC):
    """Abstract base class for parameter analyzers."""

    def __init__(self, fs: float, n_sinusoids: int):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
        """
        self.fs: float = fs
        self.n_sinusoids: int = n_sinusoids
        self.subspace_dim: int = -1
        self.est_params: SinusoidParameters | None = None

    def fit(self, signal: npt.NDArray[np.complex128]) -> Self:
        """Run the full parameter estimation process."""
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
        """Build the covariance matrix from the input signal."""
        n_samples = signal.size
        n_snapshots = n_samples - subspace_dim + 1
        hankel_matrix = hankel(signal[:subspace_dim], signal[subspace_dim - 1 :])
        _cov_matrix = (hankel_matrix @ hankel_matrix.conj().T) / n_snapshots
        cov_matrix: npt.NDArray[np.complex128] = _cov_matrix.astype(np.complex128)
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
        except np.linalg.LinAlgError:
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
