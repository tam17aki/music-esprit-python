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
from scipy.linalg import LinAlgError, hankel, pinv

from utils.data_models import (
    ComplexArray,
    FloatArray,
    NumpyComplex,
    NumpyFloat,
    SignalArray,
    SinusoidParameters,
)

from .models import AnalyzerParameters

SUBSPACE_RATIO_UPPER_BOUND = 0.5


class AnalyzerBase(ABC):
    """Abstract base class for parameter analyzers."""

    def __init__(
        self, fs: float, n_sinusoids: int, subspace_ratio: float = 1 / 3
    ) -> None:
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            subspace_ratio (float, optional):
                The ratio of the subspace dimension to the signal
                length. Must be between 0 and 0.5. Defaults to 1/3.
        """
        if not 0 < subspace_ratio <= SUBSPACE_RATIO_UPPER_BOUND:
            raise ValueError(
                "subspace_ratio must be in the range "
                + f"(0, {SUBSPACE_RATIO_UPPER_BOUND}]."
            )

        self.fs: float = fs
        self.n_sinusoids: int = n_sinusoids
        self.subspace_ratio: float = subspace_ratio
        self.subspace_dim: int = -1
        self.est_params: SinusoidParameters | None = None

    def fit(self, signal: SignalArray) -> Self:
        """Run the full parameter estimation process.

        This method takes an input signal, runs the complete estimation
        workflow defined by the specific analyzer subclass, and stores
        the results in the `est_params` attribute.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            Self: The analyzer instance itself (for method chaining).
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
            warnings.warn("Invalid subspace dimension.")
            return self
        freqs = self._estimate_frequencies(signal)
        if freqs.size != self.n_sinusoids:
            self.est_params = SinusoidParameters(
                freqs, np.array([]), np.array([])
            )
            warnings.warn(
                f"Expected {self.n_sinusoids} components, "
                + f"but found {freqs.size}."
            )
            return self
        amps, phases = self._estimate_amplitudes_phases(signal, freqs)
        self.est_params = SinusoidParameters(freqs, amps, phases)
        return self

    @abstractmethod
    def _estimate_frequencies(self, signal: SignalArray) -> FloatArray:
        """Estimate frequencies using a specific MUSIC variant."""
        raise NotImplementedError

    @staticmethod
    def _build_hankel_matrix(
        signal: SignalArray, subspace_dim: int
    ) -> FloatArray | ComplexArray:
        """Build the Hankel data matrix.

        Args:
            signal (SignalArray): Input signal.
            subspace_dim (int): The dimension of subspace.

        Returns:
            FloatArray | ComplexArray: The Hankel matrix.
        """
        hankel_matrix = hankel(
            signal[:subspace_dim], signal[subspace_dim - 1 :]
        )
        if np.isrealobj(signal):
            return hankel_matrix.astype(NumpyFloat)
        return hankel_matrix.astype(NumpyComplex)

    @staticmethod
    def _build_covariance_matrix(
        signal: SignalArray, subspace_dim: int
    ) -> FloatArray | ComplexArray:
        """Build the covariance matrix from the input signal.

        Args:
            signal (SignalArray): Input signal.
            subspace_dim (int): The dimension of subspace.

        Returns:
            FloatArray | ComplexArray: The covariance matrix.
        """
        n_samples = signal.size
        n_snapshots = n_samples - subspace_dim + 1
        hankel_matrix = hankel(
            signal[:subspace_dim], signal[subspace_dim - 1 :]
        )
        cov_matrix = (hankel_matrix @ hankel_matrix.conj().T) / n_snapshots
        if np.isrealobj(signal):
            return cov_matrix.astype(NumpyFloat)
        return cov_matrix.astype(NumpyComplex)

    @staticmethod
    def _build_vandermonde_matrix(
        freqs: FloatArray, n_rows: int, fs: float
    ) -> ComplexArray:
        """Build a Vandermonde matrix from a set of frequencies."""
        # Create the time vector t as a column vector
        t_vector = np.arange(n_rows).reshape(-1, 1) / fs

        # Create the frequency vector freqs as a row vector
        freq_vector = freqs.reshape(1, -1)

        vandermonde_matrix = np.exp(2j * np.pi * t_vector @ freq_vector)
        return vandermonde_matrix.astype(np.complex128)

    def _estimate_amplitudes_phases(
        self, signal: SignalArray, estimated_freqs: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Estimate amplitudes and phases from frequencies using LS.

        Args:
            signal (SignalArray): Input signal.
            estimated_freqs (FloatArray):
                An array of estimated frequencies in Hz.

        Returns:
            tuple[FloatArray, FloatArray]:
                - estimated_amps: Estimated amplitudes.
                - estimated_phases: Estimated phases in rad.
        """
        # 1. Build the Vandermonde matrix V
        vandermonde_matrix = self._build_vandermonde_matrix(
            estimated_freqs, signal.size, self.fs
        )

        # 2. Solve for complex amplitudes c using pseudo-inverse
        # y = V @ c  =>  c = pinv(V) @ y
        try:
            complex_amps = pinv(vandermonde_matrix) @ signal
        except LinAlgError:
            warnings.warn(
                "Least squares estimation for amplitudes/phases failed."
            )
            return np.array([]), np.array([])

        # 3. Extract amplitudes and phases
        estimated_amps = np.abs(complex_amps).astype(NumpyFloat)
        if np.isrealobj(signal):
            # For a real-valued sinusoid A*cos(2*pi*f*t + phi), the
            # complex amplitude estimated using only the positive
            # frequency is (A/2)*exp(j*phi). Therefore, we need to
            # multiply the magnitude by 2.
            estimated_amps *= 2
        estimated_phases = np.angle(complex_amps).astype(NumpyFloat)

        # Sort results according to frequency for consistent comparison
        sort_indices = np.argsort(estimated_freqs)

        return estimated_amps[sort_indices], estimated_phases[sort_indices]

    def get_params(self) -> AnalyzerParameters:
        """Return a dictionary of the analyzer's hyperparameters.

        This method provides a standardized way to inspect the
        configuration of an analyzer instance. Subclasses should
        override this method to add their specific parameters.

        Returns:
            AnalyzerParameters:
                A TypedDict containing the common hyperparameters. At
                minimum, this includes 'subspace_ratio'.
        """
        return AnalyzerParameters(subspace_ratio=self.subspace_ratio)

    @property
    def frequencies(self) -> FloatArray:
        """Return the estimated frequencies in Hz after fitting."""
        if self.est_params is None:
            raise AttributeError(
                "Cannot access 'frequencies' before running fit()."
            )
        return self.est_params.frequencies

    @property
    def amplitudes(self) -> FloatArray:
        """Return the estimated amplitudes after fitting."""
        if self.est_params is None or self.est_params.amplitudes.size == 0:
            raise AttributeError(
                "Cannot access 'amplitudes' before fitting is complete."
            )
        return self.est_params.amplitudes

    @property
    def phases(self) -> FloatArray:
        """Return the estimated phases in radians after fitting."""
        if self.est_params is None or self.est_params.phases.size == 0:
            raise AttributeError(
                "Cannot access 'phases' before fitting is complete."
            )
        return self.est_params.phases
