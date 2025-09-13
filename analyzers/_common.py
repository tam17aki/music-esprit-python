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


def find_peaks_from_spectrum(
    spectrum: npt.NDArray[np.float64],
    n_sinusoids: int,
    freq_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Find the N strongest peaks from the spectrum.

    Args:
       spectrum (np.ndarray): Pseudospectrum (float64).
       n_sinusoids (int): Number of sinusoids.
       freq_grid (np.ndarray): Frequency grid (float64).

    Returns:
        np.ndarray: Strogest peaks from the spectrum.
    """
    # 1. Find all "local maxima" as peak candidates.
    #    Ignores extremely small noise floor fluctuations.
    all_peaks, _ = find_peaks(
        spectrum, height=np.median(spectrum), prominence=np.std(spectrum) / 2.0
    )
    all_peaks = np.array(all_peaks, dtype=np.int64)
    if all_peaks.size < n_sinusoids:
        return freq_grid[all_peaks] if all_peaks.size > 0 else np.array([])

    # 2. From all the peak candidates found, select N peaks
    #    with the highest spectral values.
    strongest_peak_indices = all_peaks[np.argsort(spectrum[all_peaks])[-n_sinusoids:]]
    estimated_freqs = freq_grid[strongest_peak_indices]

    return np.sort(estimated_freqs)


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
        return raw_freqs

    unique_freqs: list[npt.NDArray[np.float64]] = []
    for raw_freq in raw_freqs:
        if any(np.abs(raw_freq - _freq) <= TOLERANCE_LEVEL for _freq in unique_freqs):
            continue
        unique_freqs.append(raw_freq)

    return np.sort(np.array(unique_freqs[:n_sinusoids]))


def find_freqs_from_roots(
    coefficients: npt.NDArray[np.complex128], fs: float, n_sinusoids: int
) -> npt.NDArray[np.float64]:
    """Find roots of the polynomial and estimate frequencies.

    Args:
        coefficients (np.ndarray): The polynomial coefficients (complex128).
        fs (float): Sampling frequency in Hz.
        n_sinusoids (int): Number of sinusoids.

    Returns:
        np.ndarray: An array of estimated frequencies in Hz.
    """
    # 1. Calculate the roots of a polynomial
    try:
        roots = poly.polyroots(coefficients[::-1])  # very slow
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
