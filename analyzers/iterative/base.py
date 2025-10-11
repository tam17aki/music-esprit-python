# -*- coding: utf-8 -*-
"""Defines IterativeAnalyzerBase class for iterative greedy analyzers.

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
from typing import override

import numpy as np

from utils.data_models import (
    ComplexArray,
    FloatArray,
    NumpyComplex,
    SignalArray,
    SingleSinusoidParameters,
)

from .._common import ZERO_LEVEL
from ..base import AnalyzerBase


class IterativeAnalyzerBase(AnalyzerBase, ABC):
    """Abstract base class for iterative greedy analyzers."""

    def __init__(self, fs: float, n_sinusoids: int):
        """Initialize the analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids to estimate.
        """
        super().__init__(fs, n_sinusoids)
        self._is_real_signal: bool = True

    @override
    def _estimate_frequencies(self, signal: SignalArray) -> FloatArray:
        """Estimate frequencies by iteratively applying interpolation.

        This method follows the structure of the RELAX algorithm but
        replaces the core frequency estimation step with a call to a
        high-accuracy DFT interpolator.

        Args:
            signal (SignalArray): Input signal.

        Returns:
            FloatArray: A sorted array of estimated frequencies in Hz.
        """
        self._is_real_signal = np.isrealobj(signal)
        residual_signal = signal.copy().astype(NumpyComplex)
        estimated_freqs: list[float] = []

        for _ in range(self.n_sinusoids):
            # 1. Estimate the strongest frequency from the current
            #    residual. This is the part that subclasses will
            #    implement.
            strongest_freq = self._estimate_single_frequency(residual_signal)
            if strongest_freq is None:
                break
            estimated_freqs.append(strongest_freq)

            # 2. Estimate amplitude and phase for this frequency.
            amp, phase = self._estimate_amp_phase_single(
                residual_signal, strongest_freq
            )

            # 3. Synthesize and subtract the estimated component.
            params = SingleSinusoidParameters(strongest_freq, amp, phase)
            single_sinusoid = self._synthesize_single_sinusoid(
                params, signal.size
            )
            residual_signal -= single_sinusoid

        return np.sort(np.array(estimated_freqs))

    @abstractmethod
    def _estimate_single_frequency(self, signal: ComplexArray) -> float | None:
        """Estimate a single strongest frequency from a signal.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _estimate_amp_phase_single(
        self, signal: ComplexArray, freq: float
    ) -> tuple[float, float]:
        """Estimate amp/phase for a single sinusoid using Least Squares.

        This method solves a least-squares problem to find the best fit
        of a single sinusoid at the given frequency to the signal.

        Args:
            signal (SignalArray):
                The input signal.
            freq (float):
                The frequency of the component to estimate, in Hz.

        Returns:
            tuple[float, float]:
                A tuple containing the estimated (amplitude, phase in
                radians).
        """
        n_samples = signal.size
        t_vector = np.arange(n_samples) / self.fs
        steering_vector = np.exp(1j * 2 * np.pi * freq * t_vector)
        a_h_a = np.vdot(steering_vector, steering_vector)
        if abs(a_h_a) < ZERO_LEVEL:
            return 0.0, 0.0
        a_h_x = np.vdot(steering_vector, signal)
        complex_amp = a_h_x / a_h_a
        amp = float(np.abs(complex_amp))
        phase = float(np.angle(complex_amp))
        return amp, phase

    def _synthesize_single_sinusoid(
        self, params: SingleSinusoidParameters, n_samples: int
    ) -> ComplexArray:
        """Synthesize a single complex sinusoid from its parameters.

        This helper method generates a pure, noiseless complex-valued
        sinusoidal signal based on a given set of frequency, amplitude,
        and phase parameters. The resulting signal is used to subtract
        the estimated component from the residual in the iterative
        cancellation process.

        Args:
            params (SingleSinusoidParameters):
                A dataclass object containing the frequency, amplitude,
                and phase of the sinusoid to be synthesized.
            n_samples (int):
                The number of samples (length) of the synthesized
                signal.

        Returns:
            ComplexArray:
                The synthesized complex-`valued sinusoidal signal.
        """
        t = np.arange(n_samples) / self.fs
        argument = 2 * np.pi * params.frequency * t + params.phase
        complex_sinusoid: ComplexArray = (
            params.amplitude * np.exp(1j * argument)
        ).astype(NumpyComplex)
        return complex_sinusoid
