# -*- coding: utf-8 -*-
"""A demonstration script for comparing iterative greedy algorithms.

This script runs a comparative analysis of:
- RELAX
- CFH (with HAQSE and Candan interpolators)
- NOMP (Newtonized Orthogonal Matching Pursuit)

It evaluates their performance in terms of accuracy and speed for
estimating sinusoidal parameters.

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

from analyzers.factory import get_iterative_greedy_analyzers
from analyzers.iterative.cfh import CfhAnalyzer
from cli import (
    parse_args,
    print_experiment_setup,
    print_summary_table,
    run_and_evaluate_analyzer,
)
from utils.data_models import AlgorithmConfig, ExperimentConfig, NumpyFloat
from utils.signal_generator import create_true_parameters, generate_test_signal


def main() -> None:
    """Perform the main demonstration workflow."""
    # --- 1. Setup Configuration ---
    args = parse_args()
    config = ExperimentConfig(
        fs=args.fs,
        duration=args.duration,
        snr_db=args.snr_db,
        freqs_true=np.array(args.freqs_true, dtype=NumpyFloat),
        amp_range=tuple(args.amp_range),
    )

    # --- 2. Generate Test Signal ---
    true_params = create_true_parameters(config)
    noisy_signal = generate_test_signal(
        config.fs,
        config.duration,
        config.snr_db,
        true_params,
        is_complex=args.complex,
    )
    algo_config = AlgorithmConfig(
        subspace_ratio=args.subspace_ratio,
        n_grids=args.n_grids,
        min_freq_period=args.min_freq_period,
        ar_order=args.ar_order,
        rank_factor=args.rank_factor,
        n_fft_iip=args.n_fft_iip,
        cfh_interpolator=args.cfh_interpolator,
        n_newton_steps=args.n_newton_steps,
        n_cyclic_rounds=args.n_cyclic_rounds,
    )

    # --- 3. Build Analyzer Dictionary ---
    analyzers_to_test = get_iterative_greedy_analyzers(config, algo_config)

    # --- 4. Print Setup and Run Analyses ---
    print_experiment_setup(config, true_params)

    print("\n--- Warming up CPU and caches... ---")
    warmup_analyzer = CfhAnalyzer(fs=config.fs, n_sinusoids=config.n_sinusoids)
    warmup_analyzer.fit(noisy_signal)

    results_summary: list[dict[str, str | float]] = []
    for name, analyzer in analyzers_to_test.items():
        summary_row = run_and_evaluate_analyzer(
            name, analyzer, noisy_signal, true_params
        )
        if summary_row is not None:
            results_summary.append(summary_row)

    # --- 5. Print Summary Table ---
    print_summary_table(results_summary)


if __name__ == "__main__":
    main()
