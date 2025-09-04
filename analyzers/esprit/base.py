# -*- coding: utf-8 -*-
"""Defines EspritAnalyzerBase class for ESPRIT-based parameter analyzers.

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
from typing import override

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from ..base import AnalyzerBase


class EspritAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for ESPRIT-based parameter analyzers."""

    def __init__(self, fs: float, n_sinusoids: int, sep_factor: float = 0.4):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
        """
        super().__init__(fs, n_sinusoids)
        self.sep_factor: float = sep_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids.

        Args:
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        if self.n_sinusoids == 0:
            return np.array([])

        # 1. Estimate raw parameters
        raw_freqs = self._estimate_raw_esprit_parameters(signal)
        if raw_freqs.size == 0:
            warnings.warn("No valid parameters estimated during raw estimation.")
            return np.array([])

        # 2. Filter unique components
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        est_freqs = self._filter_unique_parameters(raw_freqs, min_separation_hz)
        return est_freqs

    def _estimate_signal_subspace(
        self, cov_matrix: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | None:
        """Estimate the signal subspace using eigenvalue decomposition."""
        model_order = 2 * self.n_sinusoids
        try:
            _, eigenvectors = eigh(cov_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Eigenvalue decomposition on covariance matrix failed.")
            return None
        _subspace = eigenvectors[:, -model_order:]
        signal_subspace: npt.NDArray[np.complex128] = _subspace.astype(np.complex128)
        return signal_subspace

    @abstractmethod
    def _solve_params_from_subspace(
        self, signal_subspace: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Solve for frequencies from the signal subspace."""
        raise NotImplementedError

    def _estimate_raw_esprit_parameters(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Perform core ESPRIT estimation to get raw parameters without filtering."""
        n_samples = signal.size
        model_order = 2 * self.n_sinusoids
        self.subspace_dim: int = n_samples // 3
        if (
            self.subspace_dim <= model_order
            or self.subspace_dim >= n_samples - model_order
        ):
            "Invalid subspace dimension for ESPRIT. Returning empty result."
            return np.array([])

        # 1. Build the covariance matrix
        cov_matrix = self._build_covariance_matrix(signal, self.subspace_dim)

        # 2. Estimate the signal subspace
        signal_subspace = self._estimate_signal_subspace(cov_matrix)
        if signal_subspace is None:
            return np.array([])

        # 3. Find frequencies from subspace
        raw_freqs = self._solve_params_from_subspace(signal_subspace)
        return raw_freqs

    def _filter_unique_parameters(
        self,
        raw_freqs: npt.NDArray[np.float64],
        min_separation_hz: float,
    ) -> npt.NDArray[np.float64]:
        """Filter raw parameters to keep a specified number of unique components."""
        if raw_freqs.size == 0:
            return np.array([])

        if raw_freqs.size <= self.n_sinusoids and min_separation_hz <= 0:
            return raw_freqs

        params = sorted(raw_freqs)
        unique_freqs = [params[0]]
        for freq in params[1:]:
            if np.abs(freq - unique_freqs[-1]) > min_separation_hz:
                unique_freqs.append(freq)

        # Limit to the number of requested sinusoids and unpack the results
        final_freqs = unique_freqs[: self.n_sinusoids]
        if not final_freqs:
            return np.array([])

        return np.array(final_freqs)
