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

from .._common import filter_unique_freqs
from ..base import AnalyzerBase


class EspritAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for ESPRIT-based parameter analyzers."""

    @abstractmethod
    def _estimate_signal_subspace(
        self, signal: npt.NDArray[np.complex128] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.float64] | None:
        """Estimate the signal subspace using eigenvalue decomposition."""
        raise NotImplementedError

    def _postprocess_omegas(
        self, raw_omegas: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Post-processes raw angular frequencies to final frequency estimates.

        This method performs a series of finalization steps:
            1. Converts normalized angular frequencies (omegas) to
               physical frequencies (Hz).
            2. Takes the positive frequencies from positive/negative frequency pairs.
            3. Sorts the frequencies in ascending order.
            4. Filters out closely spaced frequencies to return unique components.

        Args:
            raw_omegas (np.ndarray):
                An array of raw normalized angular frequencies in radians per sample,
                as returned by a solver (float64).

        Returns:
            np.ndarray:
                A sorted array of final, unique frequency estimates in Hz,
                limited to `self.n_sinusoids` (float64).
        """
        # 1. Convert normalized angular frequencies [rad/sample]
        #    to physical frequencies [Hz]
        estimated_freqs_hz = raw_omegas * (self.fs / (2 * np.pi))

        # 2. Take only the positive frequencies from positive/negative frequency pairs
        positive_freq_indices = np.where(estimated_freqs_hz > 0)[0]
        positive_freqs = estimated_freqs_hz[positive_freq_indices]

        # 3. Sort the frequencies in ascending order
        sorted_indices = np.argsort(positive_freqs)
        raw_freqs = positive_freqs[sorted_indices]

        # 4. Filters out closely spaced frequencies
        est_freqs = filter_unique_freqs(raw_freqs, self.n_sinusoids)

        return est_freqs
