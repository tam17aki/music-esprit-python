# -*- coding: utf-8 -*-
"""Defines base classes for ESPRIT-based parameter analyzers.

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

from utils.data_models import ComplexArray, FloatArray, SignalArray

from ..base import AnalyzerBase


class EspritAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for ESPRIT-based parameter analyzers."""

    def _postprocess_omegas(self, raw_omegas: FloatArray) -> FloatArray:
        """Finalizes frequency estimates from raw angular frequencies.

        This method performs a series of finalization steps:
            1. Convert angular frequencies from rad/sample to Hz.
            2. Selects positive frequency components, resolving the +/-
               pairs that occur with real-valued signals.
            3. Sort frequencies in ascending order.

        Args:
            raw_omegas (FloatArray):
                An array of raw normalized angular frequencies in
                radians per sample, as returned by a solver.

        Returns:
            FloatArray:
                A sorted array of final, unique frequency estimates in
                Hz, limited to `self.n_sinusoids`.
        """
        # 1. Convert normalized angular frequencies [rad/sample]
        #    to physical frequencies [Hz]
        estimated_freqs_hz = raw_omegas * (self.fs / (2 * np.pi))

        # 2. Extracts positive frequencies (handling +/- pairs from real
        #    signals).
        positive_freq_indices = np.where(estimated_freqs_hz > 0)[0]
        positive_freqs = estimated_freqs_hz[positive_freq_indices]

        # 3. Sort the frequencies in ascending order.
        est_freqs = np.sort(positive_freqs)

        return est_freqs


class EVDBasedEspritAnalyzer(EspritAnalyzerBase, ABC):
    """Base class for EVD/SVD-based ESPRIT variants."""

    @abstractmethod
    def _estimate_signal_subspace(
        self, signal: SignalArray
    ) -> FloatArray | ComplexArray | None:
        """Estimate the signal subspace via EVD/SVD."""
        raise NotImplementedError
