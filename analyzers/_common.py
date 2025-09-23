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


def _parabolic_interpolation(
    spectrum: npt.NDArray[np.float64], peak_indices: npt.NDArray[np.int_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Refines peak locations and magnitudes using parabolic interpolation.

    Args:
        spectrum (np.ndarray):
            The input pseudospectrum (float64).
        peak_indices (np.ndarray):
            The integer index of the peak in the spectrum.

    Returns:
        tuple[np.ndarray, np.ndarray]
            - refined_indices (np.ndarray): The refined peak locations.
            - refined_mags (np.ndarray): The refined magnitudes.
    """
    refined_indices = np.zeros_like(peak_indices, dtype=np.float64)
    refined_mags = np.zeros_like(peak_indices, dtype=np.float64)

    for i, idx in enumerate(peak_indices):
        if idx in (0, len(spectrum) - 1):
            refined_indices[i] = idx
            refined_mags[i] = spectrum[idx]
            continue
        y_minus_1 = spectrum[idx - 1]
        y_0 = spectrum[idx]
        y_plus_1 = spectrum[idx + 1]
        p = 0.5 * (y_minus_1 - y_plus_1) / (y_minus_1 - 2 * y_0 + y_plus_1)
        refined_indices[i] = idx + p
        refined_mags[i] = y_0 - 0.25 * (y_minus_1 - y_plus_1) * p

    return refined_indices, refined_mags


def find_peak_indices_from_spectrum(
    spectrum: npt.NDArray[np.float64], n_peaks: int
) -> npt.NDArray[np.int_]:
    """Finds the indices of the N strongest peaks from a spectrum.

    This function implements a robust two-stage peak finding strategy.

    Args:
        spectrum (np.ndarray):
            The input pseudospectrum (e.g., from MUSIC) (float64).
        n_peaks (int):
            The number of peaks to find and return.

    Returns:
        np.ndarray: The indices of the top-N strongest peaks.
    """
    prominence_thresh = (np.max(spectrum) - np.min(spectrum)) * 0.01
    all_peaks, _ = find_peaks(spectrum, prominence=prominence_thresh)
    if all_peaks.size == 0:
        warnings.warn("No peaks found. Falling back to simple argsort.")
        return np.argsort(spectrum)[::-1][:n_peaks]
    if all_peaks.size < n_peaks:
        warnings.warn(
            f"Found only {all_peaks.size} peaks, < {n_peaks}. "
            + "Falling back to simple argsort."
        )
        return np.argsort(spectrum)[::-1][:n_peaks]
    peak_heights = spectrum[all_peaks]
    strongest_local_indices = np.argsort(peak_heights)[::-1][:n_peaks]
    return np.sort(all_peaks[strongest_local_indices])


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
    # Find the indices of the N strongest peaks from a spectrum
    strongest_peak_indices = find_peak_indices_from_spectrum(spectrum, n_peaks)

    if strongest_peak_indices.size == 0:
        warnings.warn("No peaks found from spectrum.")
        return np.array([])

    if not use_interpolation:
        return np.sort(freq_grid[strongest_peak_indices])

    # Interpolation allows for more accurate peak position (index)
    refined_indices, _ = _parabolic_interpolation(spectrum, strongest_peak_indices)

    # Calculate the final frequency from the interpolated index
    # (freq_grid[1] - freq_grid[0]) is frequency bin width
    freq_resolution = freq_grid[1] - freq_grid[0]
    estimated_freqs = refined_indices * freq_resolution

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
