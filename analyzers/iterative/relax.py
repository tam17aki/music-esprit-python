# -*- coding: utf-8 -*-
"""Defines RelaxAnalyzer class for the RELAX algorithm.

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

from typing import final, override

from utils.data_models import ComplexArray

from .._common import estimate_freqs_iterative_fft
from .base import IterativeAnalyzerBase


@final
class RelaxAnalyzer(IterativeAnalyzerBase):
    """Analyzes sinusoidal parameters using the RELAX algorithm."""

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        n_fft_iip: int | None = None,
    ):
        """Initialize the analyzer with an experiment configuration.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids.
            n_fft_iip (int | None, optional):
                The length of iterative interpolation FFT.
                Defaults to None.
        """
        super().__init__(fs, n_sinusoids)
        self.n_fft_iip = n_fft_iip

    @override
    def _estimate_single_frequency(self, signal: ComplexArray) -> float | None:
        """Estimate the strongest frequency using a zero-padded FFT.

        Args:
            signal (SignalArray): The input signal.

        Returns:
            float: The strongest frequency in the input signal.
                Returns None on failure.
        """
        est_freqs = estimate_freqs_iterative_fft(
            signal, n_peaks=1, fs=self.fs, n_fft=self.n_fft_iip
        )
        return est_freqs[0] if len(est_freqs) > 0 else None
