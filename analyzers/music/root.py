# -*- coding: utf-8 -*-
"""Defines RootMusicAnalyzer class for Root MUSIC.

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

from mixins.covariance import ForwardBackwardMixin

from .._common import find_freqs_from_roots
from .base import MusicAnalyzerBase


class RootMusicAnalyzer(MusicAnalyzerBase):
    """MUSIC analyzer using polynomial rooting."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        sep_factor: float = 0.4,
        subspace_ratio: float = 1 / 3,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            sep_factor (float, optional):
                Separation factor for resolving close frequencies.
            subspace_ratio (float, optional): The ratio of the subspace dimension
                to the signal length. Should be between 0 and 0.5. Defaults to 1/3.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio)
        self.sep_factor: float = sep_factor

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate frequencies of multiple sinusoids using Root MUSIC.

        Args:
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        # 1. Estimate the noise subspace
        noise_subspace = self._estimate_noise_subspace(signal)
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace.")
            return np.array([])

        # 2. Calculates the coefficients of the Root MUSIC polynomial
        coefficients = self._calculate_polynomial_coefficients(noise_subspace)

        # 3. Find the roots and estimate the frequencies
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        estimated_freqs = find_freqs_from_roots(
            self.fs, self.n_sinusoids, coefficients, min_separation_hz
        )

        return estimated_freqs

    @staticmethod
    def _calculate_polynomial_coefficients(
        noise_subspace: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        """Calculate the coefficients of the Root MUSIC polynomial D(z).

        Args:
            noise_subspace (np.ndarray): The noise subspace matrix E_n (complex128).

        Returns:
            np.ndarray: A vector of polynomial coefficients (float64).
        """
        # C = E_n * E_n^H
        projector_onto_noise = noise_subspace @ noise_subspace.conj().T

        # The order of denominator polynomial (L-1)
        # L = subspace_dim
        subspace_dim = projector_onto_noise.shape[0]
        poly_degree = subspace_dim - 1

        coefficients = np.zeros(2 * subspace_dim - 1, dtype=np.complex128)
        for k in range(-poly_degree, poly_degree + 1):
            # Computes the sum of the k-th diagonal of matrix C
            diag_sum = np.sum(np.diag(projector_onto_noise, k=k))
            coefficients[k + poly_degree] = diag_sum

        # Notice: The polynomial coefficients are arranged in descending order of powers
        return coefficients


@final
class RootMusicAnalyzerFB(ForwardBackwardMixin, RootMusicAnalyzer):
    """Root MUSIC analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
