# -*- coding: utf-8 -*-
"""A collection of functions for building test signals.

This module provides the complete pipeline for generating test signals,
from creating the ground truth parameters to synthesizing the clean
waveform and adding controlled noise. These functions are used to create
the datasets for demonstrating and evaluating the analyzer algorithms.

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

import numpy as np
import numpy.typing as npt

from .data_models import ExperimentConfig, SinusoidParameters


def _generate_amps_phases(
    amp_range: tuple[float, float],
    n_sinusoids: int,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate amplitudes and phases for multiple sinusoids.

    Args:
        amp_range (tuple[float, float]):
            Lower and upper bound for amplitude.
        n_sinusoids (int): Number of sinusoids.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - amps: An array of random amplitudes (float64).
            - phases: An array of random phases (float64).
    """
    if rng is None:
        rng = np.random.default_rng()
    amps = rng.uniform(amp_range[0], amp_range[1], n_sinusoids).astype(np.float64)
    phases = rng.uniform(-np.pi, np.pi, n_sinusoids).astype(np.float64)
    return amps, phases


def create_true_parameters(
    config: ExperimentConfig, rng: np.random.Generator | None = None
) -> SinusoidParameters:
    """Create a SinusoidParameters object with true values.

    Args:
        config (ExperimentConfig): Configuration of the experiment.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        SinusoidParameters: An object containing the true signal params.
    """
    # Generate the random parts of the parameters
    amps_true, phases_true = _generate_amps_phases(
        config.amp_range, config.n_sinusoids, rng
    )

    # Combine fixed parts (from config) and random parts
    true_params = SinusoidParameters(
        frequencies=config.freqs_true, amplitudes=amps_true, phases=phases_true
    )
    return true_params


def synthesize_sinusoids(
    fs: float,
    duration: float,
    params: SinusoidParameters,
    *,
    is_complex: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Synthesize a clean signal from multiple sinusoids.

    Args:
        fs (float): Sampling frequency in Hz.
        duration (float): Signal duration in seconds.
        params (SinusoidParameters): Parametes of mutiple sinusoids.
        is_complex (bool, optional):
            If True, generate a complex exponential signal. If False,
            generate a real-valued cosine signal. Defaults to False.

    Returns:
        np.ndarray: Sum of multiple sinusoids (float64 or complex128).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False).reshape(1, -1)
    freqs = params.frequencies.reshape(-1, 1)
    amps = params.amplitudes.reshape(-1, 1)
    phases = params.phases.reshape(-1, 1)
    argument = 2 * np.pi * freqs @ t + phases
    if is_complex:
        clean_signal_complex: npt.NDArray[np.complex128]
        clean_signal_complex = np.sum(amps * np.exp(1j * argument), axis=0)
        return clean_signal_complex
    clean_signal_real: npt.NDArray[np.float64]
    clean_signal_real = np.sum(amps * np.cos(argument), axis=0)
    return clean_signal_real


def add_awgn(
    signal: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Add Additive White Gaussian Noise (AWGN) to a given signal.

    Args:
        signal (np.ndarray): Input clean signal (float64 or complex128).
        snr_db (float): Target signal-to-noise ratio in dB.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        np.ndarray:
            Noisy signal with specified SNR (float64 or complex128).
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), signal.size)
    return signal + noise


def generate_test_signal(
    fs: float,
    duration: float,
    snr_db: float,
    params: SinusoidParameters,
    *,
    is_complex: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Generate a noisy test signal of multiple sinusoids.

    This is a convenience wrapper that combines sinusoid synthesis and
    noise addition into a single step.

    Args:
        fs (float): Sampling frequency in Hz.
        duration (float): Signal duration in seconds.
        snr_db (float): Target signal-to-noise ratio in dB.
        params (SinusoidParameters): Parametes of mutiple sinusoids.
        is_complex (bool, optional):
            If True, generate a complex-valued test signal.
            If False, generate a real-valued test signal.
            Defaults to False.

    Returns:
        np.ndarray:
            The generated noisy test signal (float64 or complex128),
            depending on the `is_complex` flag.
    """
    clean_signal = synthesize_sinusoids(fs, duration, params, is_complex=is_complex)
    noisy_signal = add_awgn(clean_signal, snr_db)
    return noisy_signal
