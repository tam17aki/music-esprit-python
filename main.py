# -*- coding: utf-8 -*-
"""A demonstration of parameter estimation for sinusoidal signals.

Frequencies are estimated using the MUSIC and ESPRIT algorithm, followed by
amplitude and phase estimation via the least squares method.

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

from analyzers.esprit.solvers import LSEspritSolver
from analyzers.esprit.standard import StandardEspritAnalyzer
from analyzers.music.root import RootMusicAnalyzer
from analyzers.music.spectral import SpectralMusicAnalyzer
from cli import parse_args, print_experiment_setup, print_results
from utils.data_models import ExperimentConfig
from utils.signal_generator import create_true_parameters, generate_test_signal


def main() -> None:
    """Perform demonstration."""
    args = parse_args()
    config = ExperimentConfig(
        fs=args.fs,
        duration=args.duration,
        snr_db=args.snr_db,
        freqs_true=np.array(args.freqs_true, dtype=np.float64),
        amp_range=tuple(args.amp_range),
        n_grids=args.n_grids,
        subspace_ratio=args.subspace_ratio,
    )

    # Generate test signals (sum of multiple sinusoids with additive noise)
    true_params = create_true_parameters(config)
    noisy_signal = generate_test_signal(
        config.fs,
        config.duration,
        config.snr_db,
        true_params,
        is_complex=args.complex,
    )

    # Print the experiment setup
    print_experiment_setup(config, true_params)

    # Perform parameter estimation via Spectral MUSIC
    print("\n--- Running Spectral MUSIC ---")
    spec_analyzer = SpectralMusicAnalyzer(
        config.fs,
        config.n_sinusoids,
        n_grids=config.n_grids,
        subspace_ratio=config.subspace_ratio,
    )
    spec_analyzer.fit(noisy_signal)

    # Print results
    print_results(spec_analyzer, true_params)

    # Perform parameter estimation via Root MUSIC
    print("\n--- Running Root MUSIC ---")
    root_analyzer = RootMusicAnalyzer(
        config.fs,
        config.n_sinusoids,
        subspace_ratio=config.subspace_ratio,
    )
    root_analyzer.fit(noisy_signal)

    # Print results
    print_results(root_analyzer, true_params)

    # Perform parameter estimation via ESPRIT
    print("\n--- Running ESPRIT ---")
    esprit_analyzer = StandardEspritAnalyzer(
        config.fs,
        config.n_sinusoids,
        LSEspritSolver(),
        subspace_ratio=config.subspace_ratio,
    )
    esprit_analyzer.fit(noisy_signal)

    # Print results
    print_results(esprit_analyzer, true_params)


if __name__ == "__main__":
    main()
