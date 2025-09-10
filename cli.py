# -*- coding: utf-8 -*-
"""Helper functions for command line arguments and priting results.

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

import argparse

import numpy as np

from analyzers.esprit.standard import StandardEspritAnalyzer
from analyzers.music.root import RootMusicAnalyzer
from analyzers.music.spectral import SpectralMusicAnalyzer
from utils.data_models import ExperimentConfig, SinusoidParameters


def print_experiment_setup(
    config: ExperimentConfig, true_params: SinusoidParameters
) -> None:
    """Print the setup of the experiment."""
    sort_indices = np.argsort(true_params.frequencies)
    print("--- Experiment Setup ---")
    print(f"Sampling Frequency: {config.fs} Hz")
    print(f"Signal Duration:    {config.duration * 1000:.0f} ms")
    print(f"True Frequencies:   {true_params.frequencies[sort_indices]} Hz")
    print(f"True Amplitudes:    {true_params.amplitudes[sort_indices]}")
    print(f"True Phases:        {true_params.phases[sort_indices]} rad")
    print(f"SNR:                {config.snr_db} dB")
    print(f"Number of Grid Points:  {config.n_grids}")


def print_results(
    analyzer: SpectralMusicAnalyzer | RootMusicAnalyzer | StandardEspritAnalyzer,
    true_params: SinusoidParameters,
) -> None:
    """Print the results."""
    if analyzer.est_params is None:
        print("MusicAnalyzer is not fitted.")
        return
    if analyzer.frequencies.size != true_params.frequencies.size:
        print(
            "Estimation incomplete or failed. "
            + f"Found {analyzer.frequencies.size} components."
        )
        print(f"Est Frequencies: {analyzer.frequencies} Hz")
        return

    print("\n--- Estimation Results ---")
    print(f"Est Frequencies: {analyzer.frequencies} Hz")
    print(f"Est Amplitudes:  {analyzer.amplitudes}")
    print(f"Est Phases:      {analyzer.phases} rad")

    sort_indices = np.argsort(true_params.frequencies)
    freq_errors = analyzer.frequencies - true_params.frequencies[sort_indices]
    amp_errors = analyzer.amplitudes - true_params.amplitudes[sort_indices]
    phase_errors = analyzer.phases - true_params.phases[sort_indices]
    print("\n--- Estimation Errors ---")
    print(f"Freq Errors:  {freq_errors} Hz")
    print(f"Amp Errors:   {amp_errors}")
    print(f"Phase Errors: {phase_errors} rad\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MUSIC demo."""
    parser = argparse.ArgumentParser(
        description="Parameter estimation demo using MUSIC algorithm."
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=44100.0,
        help="Sampling frequency in Hz (default: 44100.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Signal duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=30.0,
        help="Signal-to-noise ratio in dB (default: 30.0)",
    )
    parser.add_argument(
        "--freqs_true",
        type=float,
        nargs="+",
        default=[440.0, 460.0, 480.0],
        help="List of true frequencies in Hz (space separated). "
        + "Default: 440.0 460.0 480.0",
    )
    parser.add_argument(
        "--amp_range",
        type=float,
        nargs=2,
        default=[0.5, 1.5],
        metavar=("AMP_MIN", "AMP_MAX"),
        help="Amplitude range for sinusoid generation (default: 0.5 1.5)",
    )
    parser.add_argument(
        "--n_grids",
        type=int,
        default=8192,
        help="Number of frequency grid points for MUSIC spectrum (default: 8192)",
    )
    parser.add_argument(
        "--sep_factor",
        type=float,
        default=0.4,
        help="Separation factor for resolving close frequencies, "
        + "relative to FFT resolution (fs / n_samples). "
        + "A value < 0.5 can help separate frequencies closer than the FFT limit. "
        + "(default: 0.4)",
    )
    return parser.parse_args()
