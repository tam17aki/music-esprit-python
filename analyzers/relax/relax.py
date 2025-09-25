# -*- coding: utf-8 -*-
"""Defines RelaxEspritAnalyzer class for RELAX algorithm.

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

from dataclasses import dataclass
from typing import final, override

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError

from .._common import estimate_freqs_iterative_fft
from ..base import AnalyzerBase

ZERO_LEVEL = 1e-9


@dataclass
class SingleSinusoidParameters:
    """Represents the parameters of a single sinusoid."""

    frequency: float
    amplitude: float
    phase: float


@final
class RelaxEspritAnalyzer(AnalyzerBase):
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
            n_fft_iip (int, optional): The length of iterative interpolation FFT.
        """
        super().__init__(fs, n_sinusoids, subspace_ratio=0.5)
        self.n_fft_iip = n_fft_iip

    @override
    def _estimate_frequencies(
        self, signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Estimate the signal subspace using eigenvalue decomposition.

        Args:
            signal (np.ndarray): Input signal (float64 or complex128).

        Returns:
            np.ndarray: Estimated frequencies in Hz (float64).
        """
        residual_signal = signal.copy()
        estimated_freqs: list[float] = []
        params = SingleSinusoidParameters(0.0, 0.0, 0.0)
        for _ in range(self.n_sinusoids):
            strongest_freq = estimate_freqs_iterative_fft(
                residual_signal, n_peaks=1, fs=self.fs
            )[0]
            estimated_freqs.append(strongest_freq)
            amp, phase = self._estimate_amp_phase(residual_signal, strongest_freq)
            params.frequency = float(strongest_freq)
            params.amplitude = float(amp)
            params.phase = float(phase)
            single_sinsoid = self.synthesize_single_sinusoid(
                params, signal.size, self.fs, np.isrealobj(signal)
            )
            if np.isrealobj(signal):
                sinusoid_to_remove = single_sinsoid.astype(np.float64)
            else:
                sinusoid_to_remove = single_sinsoid.astype(np.complex128)
            residual_signal -= sinusoid_to_remove

        return np.sort(np.array(estimated_freqs))

    @staticmethod
    def synthesize_single_sinusoid(
        params: SingleSinusoidParameters,
        n_samples: int,
        fs: float,
        is_real_signal: bool,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Re-synthesize a single  sinusoid from its parameter object.

        Args:
            params (SingleSinusoidParameters):
                The parameters of the single sinusoid.
            n_samples (int):
                The number of samples in the sinsoid.
            fs (float):
                Sampling frequency in Hz of the sinusoid.
            is_real_signal (bool):
                A flag indicating whether the sinusoid is real-valued.
        """
        t = np.arange(n_samples) / fs
        argument = 2 * np.pi * params.frequency * t + params.phase
        if is_real_signal:
            signal_real = params.amplitude * np.cos(argument)
            return signal_real
        signal_complex: npt.NDArray[np.complex128] = (
            params.amplitude * np.exp(1j * argument)
        ).astype(np.complex128)
        return signal_complex

    def _estimate_amp_phase(
        self,
        signal: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
        freq: float,
    ) -> tuple[float, float]:
        """Estimates the amplitude and phase of a single sinusoidal component.

        This method solves a least-squares problem to find the best fit
        of a single sinusoid at the given frequency to the signal.

        Args:
            signal (np.ndarray): The input signal (float64 or complex128).
            freq (float): The frequency of the component to estimate, in Hz.

        Returns:
            tuple[float, float]: A tuple containing the estimated
                                 (amplitude, phase in radians).
        """
        # 1. Construct a single steering vector (one column of the Vandermonde matrix)
        n_samples = signal.size
        t_vector = np.arange(n_samples).reshape(-1, 1) / self.fs

        # The steering vector has the shape (n_samples, 1)
        steering_vector = np.exp(2j * np.pi * freq * t_vector)

        # Internal calculations are done with complex numbers
        complex_signal = signal.astype(np.complex128)

        # 2. Solve complex amplitudes with least squares
        #    c = pinv(V) @ x
        #    pinv(vector) is equivalent to (vector^H * vector)^-1 * vector^H
        try:
            # For pinv calculations, using np.dot is faster
            # when there is only one vector.
            # c = (a^H * a)^-1 * a^H * x
            a_h_a = np.dot(steering_vector.conj().T, steering_vector)
            a_h_x = np.dot(steering_vector.conj().T, complex_signal)

            if np.abs(a_h_a) < ZERO_LEVEL:
                return 0.0, 0.0

            complex_amp: npt.NDArray[np.complex128] = (a_h_x / a_h_a).astype(
                np.complex128
            )

        except LinAlgError:
            return 0.0, 0.0

        # 3. Extract amplitude and phase
        amp = float(np.abs(complex_amp))
        phase = float(np.angle(complex_amp))

        # 4. If the input is a real signal, double the amplitude.
        if np.isrealobj(signal):
            amp *= 2.0

        return amp, phase
