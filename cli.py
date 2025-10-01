# -*- coding: utf-8 -*-
"""Defines the Command-Line Interface (CLI) for the demonstration script.

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
import time

import numpy as np

from analyzers.base import SUBSPACE_RATIO_UPPER_BOUND, AnalyzerBase
from utils.data_models import ExperimentConfig, SignalArray, SinusoidParameters


def print_experiment_setup(
    config: ExperimentConfig, true_params: SinusoidParameters
) -> None:
    """Print the experimental setup parameters to the console.

    This function displays the configuration of the simulation world
    (e.g., sampling frequency, SNR) and the ground truth parameters of
    the synthesized signal in a formatted table.

    Args:
        config (ExperimentConfig):
            An object containing the overall experimental configuration.
        true_params (SinusoidParameters):
            An object containing the ground truth parameters
            (frequencies, amplitudes, phases) of the signal sources.
    """
    sort_indices = np.argsort(true_params.frequencies)
    print("--- Experiment Setup ---")
    print(f"Sampling Frequency: {config.fs} Hz")
    print(f"Signal Duration:    {config.duration * 1000:.0f} ms")
    print(f"SNR:                {config.snr_db} dB")
    print(f"True Frequencies:   {true_params.frequencies[sort_indices]} Hz")
    print(f"True Amplitudes:    {true_params.amplitudes[sort_indices]}")
    print(f"True Phases:        {true_params.phases[sort_indices]} rad")


def print_analyzer_info(analyzer: AnalyzerBase) -> None:
    """Print the hyperparameters of a given analyzer instance.

    This function retrieves the configuration parameters from an
    analyzer using its `get_params()` method and displays them in a
    human-readable format. It is used to report the settings for each
    analysis run.

    Args:
        analyzer (AnalyzerBase):
            The analyzer instance whose parameters are to be
            printed. Must be an instance of a class inheriting from
            `AnalyzerBase`.
    """
    params = analyzer.get_params()
    print("Analyzer Parameters:")
    if not any(params):  # Check if the dictionary is empty
        print("  (No specific parameters to display)")
        return
    for key, value in params.items():
        formatted_key = key.replace("_", " ").title()
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        print(f"  {formatted_key}: {formatted_value}")


def print_results(
    analyzer: AnalyzerBase, true_params: SinusoidParameters
) -> None:
    """Print the estimation results and errors from a fitted analyzer.

    This function retrieves estimated parameters from a fitted analyzer
    (e.g., via its `.frequencies` property) and displays them in a
    formatted table, alongside estimation errors calculated against the
    ground truth.

    Args:
        analyzer (AnalyzerBase):
            A fitted analyzer object.
            It must be an instance of a class that inherits from
            AnalyzerBase and has been run with .fit().
        true_params (SinusoidParameters):
            An object containing the ground truth parameters for
            calculating the estimation errors.
    """
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


def compute_summary_row(
    name: str,
    analyzer: AnalyzerBase,
    true_params: SinusoidParameters,
    elapsed_time: float,
) -> dict[str, str | float] | None:
    """Compute a single row of the results summary table.

    Calculates RMSEs for frequency, amplitude, and phase from a fitted
    analyzer and returns them as a dictionary.

    Args:
        name (str): The name of the method being evaluated.
        analyzer (AnalyzerBase): The fitted analyzer instance.
        true_params (SinusoidParameters): The ground truth parameters.
        elapsed_time (float): The execution time of the .fit() method.

    Returns:
        dict[str, str | float] | None:
            A dictionary representing one row of the summary table,
            or None if the estimation was incomplete.
    """
    if (
        analyzer.est_params is None
        or analyzer.frequencies.size != true_params.frequencies.size
    ):
        return None

    sort_indices = np.argsort(true_params.frequencies)

    # Frequency Error
    freq_errors = analyzer.frequencies - true_params.frequencies[sort_indices]
    freq_rmse = np.sqrt(np.mean(freq_errors**2))

    # amplitude error
    amp_errors = analyzer.amplitudes - true_params.amplitudes[sort_indices]
    amp_rmse = np.sqrt(np.mean(amp_errors**2))

    # phase error
    phase_errors = analyzer.phases - true_params.phases[sort_indices]
    phase_rmse = np.sqrt(np.mean(phase_errors**2))

    return {
        "Method": name,
        "Time (s)": elapsed_time,
        "Freq RMSE (Hz)": freq_rmse,
        "Amp RMSE": amp_rmse,
        "Phase RMSE (rad)": phase_rmse,
    }


def run_and_evaluate_analyzer(
    name: str,
    analyzer: AnalyzerBase,
    signal: SignalArray,
    true_params: SinusoidParameters,
) -> dict[str, str | float] | None:
    """Run an analyzer, prints results, and returns a summary row.

    This function encapsulates the entire workflow for a single analysis
    run:
    1. Prints the analyzer's name and hyperparameters.
    2. Measures the execution time of the .fit() method.
    3. Prints the detailed estimation results.
    4. Computes and returns a summary row for the final table.

    Args:
        name (str): The human-readable name of the analyzer.
        analyzer (AnalyzerBase): The analyzer instance to run.
        signal (SignalArray): The input signal.
        true_params (SinusoidParameters): The ground truth parameters.

    Returns:
        dict[str, str | float] | None:
            A dictionary representing one row of the summary table,
            or None if the estimation was incomplete.
    """
    print(f"\n--- Running {name} ---")
    print_analyzer_info(analyzer)

    start_time = time.perf_counter()
    analyzer.fit(signal)
    end_time = time.perf_counter()

    print(f"Elapsed Time: {end_time - start_time:.4f} seconds")
    print_results(analyzer, true_params)

    summary_row = compute_summary_row(
        name, analyzer, true_params, end_time - start_time
    )

    return summary_row


def print_summary_table(results: list[dict[str, str | float]]) -> None:
    """Print a summary table of the estimation results.

    Args:
        results (list[dict[str, str | float]]):
            A list of result rows, where each row is a dictionary
            produced by `compute_summary_row`.
    """
    if not results:
        print("\n--- No results to summarize. ---")
        return

    print("\n--- Results Summary ---")

    # Determine the header
    headers = list(results[0].keys())

    # Calculate the maximum width for each column
    col_widths = {key: len(key) for key in headers}
    for row in results:
        for key, value in row.items():
            # Calculate the length of the formatted value
            if isinstance(value, float):
                val_len = len(f"{value:.6f}")
            else:
                val_len = len(str(value))
            col_widths[key] = max(col_widths[key], val_len)

    # --- Output header line ---
    header_line = " | ".join([f"{h:<{col_widths[h]}}" for h in headers])
    print(header_line)

    # --- Output separator lines ---
    separator_line = "-|-".join(["-" * col_widths[h] for h in headers])
    print(separator_line)

    # --- Output data rows ---
    for row in results:
        row_items: list[str] = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                formatted_value = f"{value:<{col_widths[key]}.6f}"
            else:
                formatted_value = f"{str(value):<{col_widths[key]}}"
            row_items.append(formatted_value)
        print(" | ".join(row_items))


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the demo script.

    Configures an ArgumentParser to accept settings for the signal
    synthesis (e.g., SNR, frequencies) and for the various analyzer
    algorithms (e.g., subspace ratio, number of grids).

    Returns:
        argparse.Namespace:
            An object containing the parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parameter estimation demo using MUSIC algorithm."
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=44100.0,
        help="Sampling frequency in Hz (default: 44100.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Signal duration in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=30.0,
        help="Signal-to-noise ratio in dB (default: 30.0).",
    )
    parser.add_argument(
        "--freqs_true",
        type=float,
        nargs="+",
        default=[440.0, 460.0, 480.0],
        help="List of true frequencies in Hz (space separated) "
        + "(default: 440.0 460.0 480.0).",
    )
    parser.add_argument(
        "--amp_range",
        type=float,
        nargs=2,
        default=[0.5, 1.5],
        metavar=("AMP_MIN", "AMP_MAX"),
        help="Range for random generation of sinusoidal amplitudes "
        + "(default: 0.5 1.5).",
    )
    parser.add_argument(
        "--subspace_ratio",
        type=float,
        default=1 / 3,
        help="Ratio of the subspace dimension to the signal length "
        + "(default: 1/3, which is approximately 0.333). "
        + "This value (L/N) determines the size of the covariance matrix. "
        + f"Must be in the range (0, {SUBSPACE_RATIO_UPPER_BOUND}].",
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="If specified, generate a complex-valued test signal "
        + "instead of a real-valued one.",
    )
    parser.add_argument(
        "--n_grids",
        type=int,
        default=16384,
        help="Number of frequency grid points for Spectral MUSIC and Spectral "
        + "Min-Norm method (default: 16384).",
    )
    parser.add_argument(
        "--min_freq_period",
        type=float,
        default=20.0,
        help="Minimum frequency for periodicity search for FAST MUSIC method "
        + "(default: 20.0).",
    )
    parser.add_argument(
        "--ar_order",
        type=int,
        default=512,
        help="Order of the AutoRegressive (AR) model "
        + "for HOYW method. (default: 512)",
    )
    parser.add_argument(
        "--rank_factor",
        type=int,
        default=10,
        help="Factor to determine the number of rows to sample "
        + "for Nyström-based ESPRIT method (default: 10).",
    )

    args = parser.parse_args()
    if not 0 < args.subspace_ratio <= SUBSPACE_RATIO_UPPER_BOUND:
        parser.error(
            "Argument --subspace_ratio must be in the range "
            + f"(0, {SUBSPACE_RATIO_UPPER_BOUND}]."
        )

    return args
