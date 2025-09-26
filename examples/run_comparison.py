# -*- coding: utf-8 -*-
"""A demonstration script for sinusoidal parameter estimation.

This script runs a comparative analysis of high-resolution parameter estimation
algorithms:
- MUSIC (Spectral and Root)
- ESPRIT (Standard LS)

For each method, it estimates the frequencies, amplitudes, and phases of
sinusoidal components in a noisy signal and reports the estimation errors.

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

import time

import numpy as np

from analyzers.esprit.solvers import LSEspritSolver
from analyzers.esprit.standard import StandardEspritAnalyzer
from analyzers.music.root import RootMusicAnalyzer
from analyzers.music.spectral import SpectralMusicAnalyzer
from cli import (
    compute_summary_row,
    parse_args,
    print_analyzer_info,
    print_experiment_setup,
    print_results,
    print_summary_table,
)
from utils.data_models import ExperimentConfig
from utils.signal_generator import create_true_parameters, generate_test_signal


def main() -> None:
    """Perform the main demonstration workflow."""
    # --- 1. Setup Configuration ---
    args = parse_args()
    config = ExperimentConfig(
        fs=args.fs,
        duration=args.duration,
        snr_db=args.snr_db,
        freqs_true=np.array(args.freqs_true, dtype=np.float64),
        amp_range=tuple(args.amp_range),
    )

    # --- 2. Generate Test Signal ---
    true_params = create_true_parameters(config)
    noisy_signal = generate_test_signal(
        config.fs, config.duration, config.snr_db, true_params, is_complex=args.complex
    )

    # --- 3. Print Setup and Run Analyses ---
    print_experiment_setup(config, true_params)

    analyzers_to_test = {
        "Spectral MUSIC": SpectralMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=args.n_grids,
            subspace_ratio=args.subspace_ratio,
        ),
        "Root MUSIC": RootMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=args.subspace_ratio,
        ),
        "ESPRIT (LS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
            subspace_ratio=args.subspace_ratio,
        ),
    }

    results_summary: list[dict[str, str | float]] = []
    for name, analyzer in analyzers_to_test.items():
        print(f"\n--- Running {name} ---")
        print_analyzer_info(analyzer)

        start_time = time.perf_counter()
        analyzer.fit(noisy_signal)
        end_time = time.perf_counter()

        print(f"Elapsed Time: {end_time - start_time:.4f} seconds")
        print_results(analyzer, true_params)

        # Call the function and get the resulting rows
        summary_row = compute_summary_row(
            name, analyzer, true_params, end_time - start_time
        )

        # If the result is valid, add it to the summary list
        if summary_row is not None:
            results_summary.append(summary_row)

    # --- 4. Print Summary Table ---
    print_summary_table(results_summary)


if __name__ == "__main__":
    main()
