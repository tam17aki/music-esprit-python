# -*- coding: utf-8 -*-
"""The definition of RootMusicAnalyzer class.

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
import numpy.polynomial.polynomial as poly
import numpy.typing as npt
from mixins.covariance import ForwardBackwardMixin

from .base import MusicAnalyzerBase


class RootMusicAnalyzer(MusicAnalyzerBase):
    """MUSIC analyzer using polynomial rooting."""

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
        """Estimate frequencies of multiple sinusoids using Root-MUSIC.

        Args:
            signal (np.ndarray):
                Input signal (complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
                Returns empty arrays if estimation fails.
        """
        n_samples = signal.size
        subspace_dim = n_samples // 3
        model_order = 2 * self.n_sinusoids
        if subspace_dim <= model_order:
            warnings.warn("Invalid subspace dimension for MUSIC.")
            return np.array([])

        # 1. Estimate the noise subspace
        noise_subspace = self._estimate_noise_subspace(
            signal, subspace_dim, model_order
        )
        if noise_subspace is None:
            warnings.warn("Failed to estimate noise subspace.")
            return np.array([])

        # 2. Calculates the coefficients of the root-MUSIC polynomial
        coefficients = self._calculate_polynomial_coefficients(noise_subspace)

        # 3. Find the roots and estimate the frequencies
        min_separation_hz = (self.fs / signal.size) * self.sep_factor
        estimated_freqs = self._find_roots_and_estimate_freqs(
            coefficients, min_separation_hz
        )

        return estimated_freqs

    def _calculate_polynomial_coefficients(
        self, noise_subspace: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        """Calculate the coefficients of the Root-MUSIC polynomial D(z).

        Args:
            noise_subspace (np.ndarray): The noise subspace matrix E_n.

        Returns:
            np.ndarray: A vector of polynomial coefficients.
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

        return coefficients

    def _find_roots_and_estimate_freqs(
        self,
        coefficients: npt.NDArray[np.complex128],
        min_separation_hz: float,
    ) -> npt.NDArray[np.float64]:
        """Find roots of the polynomial and estimate frequencies.

        Args:
            coefficients (np.ndarray): The polynomial coefficients.
            min_separation_hz (float): The minimum separation distance in Hz.

        Returns:
            np.ndarray: An array of estimated frequencies in Hz.
        """
        # 1. Calculate the roots of a polynomial
        try:
            roots = poly.polyroots(coefficients[::-1])
        except np.linalg.LinAlgError:
            warnings.warn("Failed to find roots of the polynomial.")
            return np.array([])

        # 2. Select the root inside the unit circle
        inside_roots = roots[np.abs(roots) < 1]
        if inside_roots.size < self.n_sinusoids:
            warnings.warn(
                f"Found only {inside_roots.size} roots inside the unit circle, "
                + f"but expected {self.n_sinusoids}."
            )
            # If it fails, choose the 4M closest to the unit circle (fallback)
            # Ideally, 2M candidates would be sufficient.
            sorted_indices = np.argsort(np.abs(np.abs(inside_roots) - 1))
            closest_roots = inside_roots[sorted_indices[: 4 * self.n_sinusoids]]
        else:
            # 2. From the inner roots, choose the 2M roots that are closest
            #    to the unit circle
            #    Ideally, M candidates would be sufficient, but since some candidates
            #    may be overlooked due to noise and numerical errors,
            #    it is recommended to secure a larger number of candidates.
            sorted_indices = np.argsort(np.abs(np.abs(inside_roots) - 1))
            closest_roots = inside_roots[sorted_indices[: 2 * self.n_sinusoids]]

        # 3. Estimate normalized angular frequency from the argument of the root
        _angles = np.angle(closest_roots)
        angles = _angles[_angles >= 0]

        # 4. Convert normalized angular frequency ω [rad/sample] to physical
        #    frequency f [Hz]
        _uniq_freqs = np.abs(angles.astype(np.float64) * (self.fs / (2 * np.pi)))

        # 5. Filter frequencies
        unique_freqs = [_uniq_freqs[0]]
        for freq in _uniq_freqs[1:]:
            if np.abs(freq - unique_freqs[-1]) > min_separation_hz:
                unique_freqs.append(freq)
        return np.sort(np.array(unique_freqs[: self.n_sinusoids]))


@final
class RootMusicAnalyzerFB(ForwardBackwardMixin, RootMusicAnalyzer):
    """Root MUSIC analyzer enhanced with Forward-Backward averaging.

    Inherits from ForwardBackwardMixin to override the covariance matrix calculation.
    """
