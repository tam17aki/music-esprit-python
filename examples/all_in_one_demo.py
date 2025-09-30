# -*- coding: utf-8 -*-
"""An all-in-one demonstration script for sinusoidal parameter estimation.

This script runs an exhaustive comparison of all implemented algorithms
and their variants available in this library, including:
- MUSIC (Spectral, Root, FAST, and FB versions)
- Min-Norm (Spectral, Root, and FB versions)
- ESPRIT (Standard, Unitary, Nyström, FFT-based, with LS/TLS/Woodbury solvers)
- HOYW (LS/TLS versions)
- RELAX

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

from analyzers.base import AnalyzerBase
from analyzers.esprit.base import EspritAnalyzerBase
from analyzers.esprit.fft import FFTEspritAnalyzer
from analyzers.esprit.nystrom import NystromEspritAnalyzer
from analyzers.esprit.solvers import (
    LSEspritSolver,
    LSUnitaryEspritSolver,
    TLSEspritSolver,
    TLSUnitaryEspritSolver,
    WoodburyLSEspritSolver,
)
from analyzers.esprit.standard import StandardEspritAnalyzer
from analyzers.esprit.unitary import UnitaryEspritAnalyzer
from analyzers.hoyw.hoyw import HoywAnalyzer
from analyzers.minnorm.base import MinNormAnalyzerBase
from analyzers.minnorm.root import RootMinNormAnalyzer, RootMinNormAnalyzerFB
from analyzers.minnorm.spectral import (
    SpectralMinNormAnalyzer,
    SpectralMinNormAnalyzerFB,
)
from analyzers.music.base import MusicAnalyzerBase
from analyzers.music.fast import FastMusicAnalyzer
from analyzers.music.root import RootMusicAnalyzer, RootMusicAnalyzerFB
from analyzers.music.spectral import (
    SpectralMusicAnalyzer,
    SpectralMusicAnalyzerFB,
)
from analyzers.relax.relax import RelaxAnalyzer
from cli import (
    compute_summary_row,
    parse_args,
    print_analyzer_info,
    print_experiment_setup,
    print_results,
    print_summary_table,
)
from utils.data_models import AlgorithmConfig, ExperimentConfig
from utils.signal_generator import create_true_parameters, generate_test_signal


def get_music_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, MusicAnalyzerBase]:
    """Factory function to create a dictionary of MUSIC analyzers.

    This function instantiates all variants of the MUSIC algorithm
    family (Spectral, Root, FAST, and their FB-enhanced versions) based
    on the provided configuration objects.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The configuration object containing algorithm-specific
            hyperparameters like `n_grids`.

    Returns:
        dict[str, MusicAnalyzerBase]:
            A dictionary mapping human-readable method names to their
            corresponding analyzer instances.

    """
    analyzers: dict[str, MusicAnalyzerBase] = {
        "Spectral MUSIC": SpectralMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
        ),
        "Spectral MUSIC FB": SpectralMusicAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
        ),
        "Root MUSIC": RootMusicAnalyzer(
            fs=config.fs, n_sinusoids=config.n_sinusoids
        ),
        "Root MUSIC FB": RootMusicAnalyzerFB(
            fs=config.fs, n_sinusoids=config.n_sinusoids
        ),
        "FAST MUSIC": FastMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
            min_freq_period=algo_config.min_freq_period,
        ),
    }
    return analyzers


def get_minnorm_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, MinNormAnalyzerBase]:
    """Factory function to create a dictionary of Min-Norm analyzers.

    This function instantiates all variants of the Min-Norm algorithm
    family (Spectral, Root, and their FB-enhanced versions) based
    on the provided configuration objects.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The configuration object containing algorithm-specific
            hyperparameters like `n_grids`.

    Returns:
        dict[str, MinNormAnalyzerBase]:
            A dictionary mapping human-readable method names to their
            corresponding analyzer instances.
    """
    analyzers: dict[str, MinNormAnalyzerBase] = {
        "Spectral Min-Norm": SpectralMinNormAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
        ),
        "Spectral Min-Norm FB": SpectralMinNormAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
        ),
        "Root Min-Norm": RootMinNormAnalyzer(
            fs=config.fs, n_sinusoids=config.n_sinusoids
        ),
        "Root Min-Norm FB": RootMinNormAnalyzerFB(
            fs=config.fs, n_sinusoids=config.n_sinusoids
        ),
    }
    return analyzers


def get_esprit_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, EspritAnalyzerBase]:
    """Factory function to create a dictionary of ESPRIT analyzers.

    This function instantiates all variants of the ESPRIT algorithm
    family (Standard, Unitary, Nyström-based & FFT-based) on the basis
    of the provided configuration objects.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The configuration object containing algorithm-specific
            hyperparameters like `n_grids`.

    Returns:
        dict[str, EspritAnalyzerBase]:
            A dictionary mapping human-readable method names to their
            corresponding analyzer instances.
    """
    analyzers: dict[str, EspritAnalyzerBase] = {
        "ESPRIT (LS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
        ),
        "ESPRIT (TLS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
        ),
        "Unitary ESPRIT (LS)": UnitaryEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSUnitaryEspritSolver(),
        ),
        "Unitary ESPRIT (TLS)": UnitaryEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSUnitaryEspritSolver(),
        ),
        "Nyström-ESPRIT (LS)": NystromEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
            nystrom_rank_factor=algo_config.rank_factor,
        ),
        "Nyström-ESPRIT (TLS)": NystromEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
            nystrom_rank_factor=algo_config.rank_factor,
        ),
        "FFT-ESPRIT (LS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
        ),
        "FFT-ESPRIT (TLS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
        ),
        "FFT-ESPRIT (Woodbury-LS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=WoodburyLSEspritSolver(),
        ),
    }
    return analyzers


def get_all_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, AnalyzerBase]:
    """Factory function to create a dictionary of all analyzers.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The configuration object containing algorithm-specific
            hyperparameters like `n_grids`.

    Returns:
        dict[str, AnalyzerBase]:
            A dictionary mapping human-readable method names to their
            corresponding analyzer instances.
    """
    analyzers: dict[str, AnalyzerBase] = {}
    analyzers.update(get_music_analyzers(config, algo_config))
    analyzers.update(get_minnorm_analyzers(config, algo_config))
    analyzers["HOYW"] = HoywAnalyzer(
        config.fs, config.n_sinusoids, ar_order=algo_config.ar_order
    )
    analyzers.update(get_esprit_analyzers(config, algo_config))
    analyzers["RELAX"] = RelaxAnalyzer(
        fs=config.fs, n_sinusoids=config.n_sinusoids
    )
    return analyzers


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
    algo_config = AlgorithmConfig(
        subspace_ratio=args.subspace_ratio,
        n_grids=args.n_grids,
        min_freq_period=args.min_freq_period,
        ar_order=args.ar_order,
        rank_factor=args.rank_factor,
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

    # --- 3. Build Analyzer Dictionary ---
    analyzers_to_test = get_all_analyzers(config, algo_config)

    # --- 4. Print Setup and Run Analyses ---
    print_experiment_setup(config, true_params)

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

    # --- 5. Print Summary Table ---
    print_summary_table(results_summary)


if __name__ == "__main__":
    main()
