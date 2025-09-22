# -*- coding: utf-8 -*-
"""Defines common helper functions.

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

import warnings

import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.typing as npt
from scipy.signal import find_peaks

TOLERANCE_LEVEL = 1e-4
ZERO_FLOOR = 1e-9


def _refine_peak_by_interpolation(
    peak_idx: np.int_,
    spectrum_db: npt.NDArray[np.float64],
    freq_grid: npt.NDArray[np.float64],
) -> np.float64:
    """Refines a single peak location using parabolic interpolation.

    Args:
        peak_idx (int): The integer index of the peak in the spectrum.
        spectrum_db (np.ndarray): The spectrum in dB scale.
        freq_grid (np.ndarray): The frequency grid corresponding to the spectrum.

    Returns:
        float64: The refined frequency estimate in Hz.
    """
    if not 0 < peak_idx < spectrum_db.size - 1:
        freq_on_grid: np.float64 = freq_grid[peak_idx]
        return freq_on_grid

    y_minus = spectrum_db[peak_idx - 1]
    y_center = spectrum_db[peak_idx]
    y_plus = spectrum_db[peak_idx + 1]

    denominator = y_minus - 2 * y_center + y_plus
    if np.abs(denominator) < ZERO_FLOOR:
        delta_idx = 0.0
    else:
        delta_idx = 0.5 * (y_minus - y_plus) / denominator

    freq_resolution = freq_grid[1] - freq_grid[0]
    refined_freq: np.float64 = freq_grid[peak_idx] + delta_idx * freq_resolution
    return refined_freq


def find_peaks_from_spectrum(
    spectrum: npt.NDArray[np.float64],
    n_peaks: int,
    freq_grid: npt.NDArray[np.float64],
    *,
    use_interpolation: bool = True,
) -> npt.NDArray[np.float64]:
    """Find the N strongest peaks from the spectrum.

    Args:
        spectrum (np.ndarray):
            The input pseudospectrum (e.g., from MUSIC) (float64).
        n_peaks (int):
            The number of peaks to find and return.
        freq_grid (np.ndarray):
            The frequency grid corresponding to the spectrum (float64).
        use_interpolation (bool, optional):
            If True, performs parabolic interpolation on the detected peaks
            to estimate their true location with sub-grid accuracy.
            If False, returns the frequencies of the grid points directly.
            Defaults to True.

    Returns:
        np.ndarray:
            A sorted array of the estimated peak frequencies in Hz (float64).
    """
    # Find all "local maxima" as peak candidates.
    # Ignores extremely small noise floor fluctuations.
    all_peaks, _ = find_peaks(
        spectrum, height=np.median(spectrum), prominence=np.std(spectrum) / 2
    )
    strongest_peak_indices: npt.NDArray[np.int_]
    if all_peaks.size < n_peaks:
        strongest_peak_indices = all_peaks
    else:
        strongest_peak_indices = all_peaks[np.argsort(spectrum[all_peaks])[-n_peaks:]]

    if strongest_peak_indices.size == 0:
        return np.array([])

    if not use_interpolation:
        return np.sort(freq_grid[strongest_peak_indices])

    spectrum_db = 10 * np.log10(spectrum + ZERO_FLOOR)
    refined_freqs = [
        _refine_peak_by_interpolation(peak_idx, spectrum_db, freq_grid)
        for peak_idx in strongest_peak_indices
    ]

    return np.sort(np.array(refined_freqs))


def filter_unique_freqs(
    raw_freqs: npt.NDArray[np.float64], n_sinusoids: int
) -> npt.NDArray[np.float64]:
    """Filter raw frequencies to keep a specified number of unique components.

    Args:
        raw_freqs (np.ndarray): The estimated raw frequencies (float64).
        n_sinusoids (int): Number of sinusoids.

    Returns:
        np.ndarray: Filtered unique frequencies (float64).
    """
    if raw_freqs.size == 0:
        warnings.warn("No raw frequencies were estimated to be filtered.")
        return np.array([])

    if raw_freqs.size <= n_sinusoids:
        return np.sort(raw_freqs)

    unique_freqs: list[npt.NDArray[np.float64]] = []
    for freq in raw_freqs:
        if any(np.abs(freq - _freq) <= TOLERANCE_LEVEL for _freq in unique_freqs):
            continue
        unique_freqs.append(freq)

    return np.sort(np.array(unique_freqs[:n_sinusoids]))


def find_freqs_from_roots(
    coefficients: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    fs: float,
    n_sinusoids: int,
) -> npt.NDArray[np.float64]:
    """Find roots of the polynomial and estimate frequencies.

    Args:
        coefficients (np.ndarray): The polynomial coefficients (float64 or complex128).
        fs (float): Sampling frequency in Hz.
        n_sinusoids (int): Number of sinusoids.

    Returns:
        np.ndarray: An array of estimated frequencies in Hz (float64).
    """
    # 1. Calculate the roots of a polynomial
    try:
        roots = poly.polyroots(coefficients[::-1])
    except np.linalg.LinAlgError:
        warnings.warn("Failed to find roots of the polynomial.")
        return np.array([])

    # 2. Select the 4M roots that are closest to the unit circle
    # Ideally, 2M candidates would be sufficient, but since some candidates may be
    # overlooked due to noise and numerical errors, it is recommended to secure a larger
    # number of candidates.
    sorted_indices = np.argsort(np.abs(np.abs(roots) - 1))
    closest_roots = roots[sorted_indices[: 4 * n_sinusoids]]

    # 3. Estimate normalized angular frequency from the argument of the root
    _angles = np.angle(closest_roots)
    angles = _angles[_angles >= 0]

    # 4. Convert normalized angular frequency ω [rad/sample] to physical
    #    frequency f [Hz]
    raw_freqs = angles.astype(np.float64) * (fs / (2 * np.pi))

    # 5. Filter frequencies
    unique_freqs = filter_unique_freqs(raw_freqs, n_sinusoids)

    return unique_freqs
