# -*- coding: utf-8 -*-
"""Defines factory functions for creating analyzer instances.

This module provides a set of factory functions that encapsulate the logic
for instantiating the various analyzer classes with the correct
configurations. Using these functions promotes code reuse and simplifies
the setup process in demonstration scripts and user applications.

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
from analyzers.esprit.standard import (
    StandardEspritAnalyzer,
    StandardEspritAnalyzerFB,
)
from analyzers.esprit.unitary import UnitaryEspritAnalyzer
from analyzers.hoyw.hoyw import HoywAnalyzer
from analyzers.iterative.cfh import CfhAnalyzer
from analyzers.iterative.nomp import NompAnalyzer
from analyzers.iterative.relax import RelaxAnalyzer
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
from utils.data_models import AlgorithmConfig, ExperimentConfig


def get_representative_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, AnalyzerBase]:
    """Create a dictionary of representative analyzer instances.

    This function instantiates a curated set of analyzers, selecting one
    representative from each major algorithm family for a high-level
    comparative overview.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            Algorithm hyperparameter configuration.

    Returns:
        dict[str, AnalyzerBase]:
            A dictionary mapping method names to their corresponding
            representative analyzer instances.
    """
    analyzers: dict[str, AnalyzerBase] = {
        "Spectral MUSIC": SpectralMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
        ),
        "Root Min-Norm": RootMinNormAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "ESPRIT (LS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
        ),
        "FFT-ESPRIT (LS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
            n_fft_iip=algo_config.n_fft_iip,
        ),
        "HOYW": HoywAnalyzer(
            config.fs,
            n_sinusoids=config.n_sinusoids,
            ar_order=algo_config.ar_order,
        ),
        "RELAX": RelaxAnalyzer(fs=config.fs, n_sinusoids=config.n_sinusoids),
    }
    return analyzers


def get_music_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, MusicAnalyzerBase]:
    """Create a dictionary of all MUSIC analyzer variants.

    This function instantiates all variants of the MUSIC algorithm
    family (Spectral, Root, FAST, and their FB-enhanced versions) based
    on the provided configuration objects.

    Args:
        config (ExperimentConfig):
            the main experiment configuration.
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
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Spectral MUSIC FB": SpectralMusicAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Root MUSIC": RootMusicAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Root MUSIC FB": RootMusicAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=algo_config.subspace_ratio,
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
    """Create a dictionary of all Min-Norm analyzer variants.

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
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Spectral Min-Norm FB": SpectralMinNormAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_grids=algo_config.n_grids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Root Min-Norm": RootMinNormAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Root Min-Norm FB": RootMinNormAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            subspace_ratio=algo_config.subspace_ratio,
        ),
    }
    return analyzers


def get_standard_esprit_variants(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, EspritAnalyzerBase]:
    """Create a dictionary of standard ESPRIT analyzer variants.

    Instantiates `StandardEspritAnalyzer` with both LS and TLS solvers.
    These represent the classic, complex-valued implementations of
    ESPRIT.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The algorithm hyperparameter configuration.

    Returns:
        dict[str, EspritAnalyzerBase]:
            A dictionary mapping method names to their corresponding
            standard ESPRIT analyzer instances.
    """
    analyzers: dict[str, EspritAnalyzerBase] = {
        "ESPRIT (LS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "ESPRIT (TLS)": StandardEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "ESPRIT FB (LS)": StandardEspritAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "ESPRIT FB (TLS)": StandardEspritAnalyzerFB(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
    }
    return analyzers


def get_unitary_esprit_variants(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, EspritAnalyzerBase]:
    """Create a dictionary of Unitary ESPRIT analyzer variants.

    Instantiates `UnitaryEspritAnalyzer` with both its LS and TLS
    solvers.  These variants operate in the real domain for improved
    computational efficiency and accuracy.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The algorithm hyperparameter configuration.

    Returns:
        dict[str, EspritAnalyzerBase]:
            A dictionary mapping method names to their corresponding
            Unitary ESPRIT analyzer instances.
    """
    analyzers: dict[str, EspritAnalyzerBase] = {
        "Unitary ESPRIT (LS)": UnitaryEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=LSUnitaryEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
        "Unitary ESPRIT (TLS)": UnitaryEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSUnitaryEspritSolver(),
            subspace_ratio=algo_config.subspace_ratio,
        ),
    }
    return analyzers


def get_fast_esprit_variants(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, EspritAnalyzerBase]:
    """Create a dictionary of Nyström-based/FFT-based ESPRIT variants.

    Instantiates `NystromEspritAnalyzer` and `FFTEspritAnalyzer`, which
    prioritize computational speed by approximating the signal subspace
    instead of computing a full EVD/SVD.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The algorithm hyperparameter configuration.

    Returns:
        dict[str, EspritAnalyzerBase]:
            A dictionary mapping method names to their corresponding
            fast ESPRIT analyzer instances.
    """
    analyzers: dict[str, EspritAnalyzerBase] = {
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
            n_fft_iip=algo_config.n_fft_iip,
        ),
        "FFT-ESPRIT (TLS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=TLSEspritSolver(),
            n_fft_iip=algo_config.n_fft_iip,
        ),
        "FFT-ESPRIT (Woodbury-LS)": FFTEspritAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            solver=WoodburyLSEspritSolver(),
            n_fft_iip=algo_config.n_fft_iip,
        ),
    }
    return analyzers


def get_esprit_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, EspritAnalyzerBase]:
    """Aggregate all implemented ESPRIT analyzer variants.

    This function aggregates all available ESPRIT variants (Standard,
    Unitary, and Fast Approximation methods) by calling their respective
    specialized factory functions.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The algorithm hyperparameter configuration.

    Returns:
        dict[str, EspritAnalyzerBase]:
            A comprehensive dictionary mapping method names to all
            corresponding ESPRIT analyzer instances.
    """
    analyzers: dict[str, EspritAnalyzerBase] = {}
    analyzers.update(get_standard_esprit_variants(config, algo_config))
    analyzers.update(get_unitary_esprit_variants(config, algo_config))
    analyzers.update(get_fast_esprit_variants(config, algo_config))
    return analyzers


def get_iterative_greedy_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, AnalyzerBase]:
    """Get a dictionary of iterative greedy analyzer instances.

    This factory function instantiates RELAX and the different variants
    of the CFH analyzer for direct comparison.

    Args:
        config (ExperimentConfig):
            The main experiment configuration.
        algo_config (AlgorithmConfig):
            The algorithm hyperparameter configuration.

    Returns:
        A dictionary mapping method names to analyzer instances.
    """
    analyzers: dict[str, AnalyzerBase] = {
        "RELAX": RelaxAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_fft_iip=algo_config.n_fft_iip,
        ),
        "CFH (HAQSE)": CfhAnalyzer(
            fs=config.fs, n_sinusoids=config.n_sinusoids, interpolator="haqse"
        ),
        "CFH (Candan)": CfhAnalyzer(
            fs=config.fs, n_sinusoids=config.n_sinusoids, interpolator="candan"
        ),
        "NOMP": NompAnalyzer(
            fs=config.fs,
            n_sinusoids=config.n_sinusoids,
            n_newton_steps=algo_config.n_newton_steps,
            n_cyclic_rounds=algo_config.n_cyclic_rounds,
        ),
    }
    return analyzers


def get_all_analyzers(
    config: ExperimentConfig, algo_config: AlgorithmConfig
) -> dict[str, AnalyzerBase]:
    """Assemble a dictionary containing all available analyzers.

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
    analyzers.update(get_iterative_greedy_analyzers(config, algo_config))
    return analyzers
