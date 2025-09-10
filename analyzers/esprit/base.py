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

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..base import AnalyzerBase


class EspritAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for ESPRIT-based parameter analyzers."""

    @abstractmethod
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.float64] | None:
        """Estimate the signal subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (complex128).

        Returns:
            np.ndarray: Estimated signal subspace matrix (complex128).
                Returns None if estimation fails.
        """
        raise NotImplementedError

    def _filter_unique_freqs(
        self, raw_freqs: npt.NDArray[np.float64], min_separation_hz: float
    ) -> npt.NDArray[np.float64]:
        """Filter raw frequencies to keep a specified number of unique components.

        Args:
            raw_freqs (np.ndarray): The estimated raw frequencies (float64).
            min_separation_hz (float): The minimum separation interval in Hz.

        Returns:
            np.ndarray: Filtered unique frequencies (float64).
        """
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
