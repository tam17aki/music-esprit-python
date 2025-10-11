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
from numpy.fft import fft, fftfreq, fftshift
from numpy.linalg import LinAlgError, pinv
from scipy.signal import find_peaks

from utils.data_models import (
    ComplexArray,
    FloatArray,
    IntArray,
    NumpyComplex,
    NumpyFloat,
    SignalArray,
)

TOLERANCE_LEVEL = 1e-4
ZERO_LEVEL = 1e-9


def _compute_parabolic_offset(
    y_minus_1: float, y_0: float, y_plus_1: float
) -> float:
    """Compute a peak's fractional offset via parabolic interpolation.

    Args:
        y_minus_1 (float): Magnitude of the sample left of the peak.
        y_0 (float): Magnitude of the peak sample.
        y_plus_1 (float): Magnitude of the sample right of the peak.

    Returns:
        float: The fractional offset `p` from the integer peak index.
               Returns 0.0 if the denominator is close to zero.
    """
    denominator = y_minus_1 - 2 * y_0 + y_plus_1
    if abs(denominator) > ZERO_LEVEL:
        return 0.5 * (y_minus_1 - y_plus_1) / denominator
    return 0.0


def _parabolic_interpolation(
    spectrum: FloatArray, peak_indices: IntArray
) -> tuple[FloatArray, FloatArray]:
    """Refine peak locations & magnitudes using parabolic interpolation.

    Args:
        spectrum (FloatArray):
            Input pseudospectrum.
        peak_indices (IntArray):
            Integer indecies of peaks in the spectrum.

    Returns:
        tuple[FloatArray, FloatArray]
            - refined_indices: The refined peak locations.
            - refined_mags: The refined magnitudes.
    """
    refined_indices = np.zeros_like(peak_indices, dtype=NumpyFloat)
    refined_mags = np.zeros_like(peak_indices, dtype=NumpyFloat)

    for i, idx in enumerate(peak_indices):
        if idx in (0, len(spectrum) - 1):
            refined_indices[i] = idx
            refined_mags[i] = spectrum[idx]
            continue
        y_m1, y_0, y_p1 = spectrum[idx - 1 : idx + 2]
        p = _compute_parabolic_offset(y_m1, y_0, y_p1)
        refined_indices[i] = idx + p
        refined_mags[i] = y_0 - 0.25 * (y_m1 - y_p1) * p

    return refined_indices, refined_mags


def find_peak_indices_from_spectrum(
    spectrum: FloatArray, n_peaks: int
) -> IntArray:
    """Find the indices of the N strongest peaks from a spectrum.

    This function uses `scipy.signal.find_peaks` with a prominence
    threshold and then selects the `n_peaks` strongest peaks from the
    detected candidates.

    Args:
        spectrum (FloatArray):
            Input pseudospectrum (e.g., from MUSIC).
        n_peaks (int):
            Number of peaks to find and return.

    Returns:
        IntArray:
            Indices of the top-N strongest peaks.
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
    spectrum: FloatArray,
    n_peaks: int,
    freq_grid: FloatArray,
    *,
    use_interpolation: bool = True,
) -> FloatArray:
    """Find the N strongest peaks from the spectrum.

    Args:
        spectrum (FloatArray):
            Input pseudospectrum (e.g., from MUSIC).
        n_peaks (int):
            Number of peaks to find and return.
        freq_grid (FloatArray):
            The frequency grid corresponding to the spectrum.
        use_interpolation (bool, optional):
            If True, performs parabolic interpolation on the detected
            peaks to estimate their true location with sub-grid
            accuracy. If False, returns the frequencies of the grid
            points directly.  Defaults to True.

    Returns:
        FloatArray:
            A sorted array of estimated peak frequencies in Hz.
    """
    # Find the indices of the N strongest peaks from a spectrum
    strongest_peak_indices = find_peak_indices_from_spectrum(spectrum, n_peaks)

    if strongest_peak_indices.size == 0:
        warnings.warn("No peaks found from spectrum.")
        return np.array([])

    if not use_interpolation:
        return np.sort(freq_grid[strongest_peak_indices])

    # Interpolation allows for more accurate peak position (index)
    refined_indices, _ = _parabolic_interpolation(
        spectrum, strongest_peak_indices
    )

    # Calculate the final frequency from the interpolated index
    # (freq_grid[1] - freq_grid[0]) is frequency bin width
    freq_resolution = freq_grid[1] - freq_grid[0]
    estimated_freqs = refined_indices * freq_resolution

    return np.sort(estimated_freqs)


def filter_unique_freqs(raw_freqs: FloatArray, n_sinusoids: int) -> FloatArray:
    """Filter frequencies to a specified number of unique values.

    This function removes closely spaced frequencies (within
    `TOLERANCE_LEVEL`) and truncates the result to `n_sinusoids`.

    Args:
        raw_freqs (FloatArray):
            The estimated raw frequencies.
        n_sinusoids (int):
            Number of sinusoids.

    Returns:
        FloatArray:
            A sorted array of filtered unique frequencies.
    """
    if raw_freqs.size == 0:
        warnings.warn("No raw frequencies were estimated to be filtered.")
        return np.array([])

    if raw_freqs.size <= n_sinusoids:
        return np.sort(raw_freqs)

    unique_freqs: list[FloatArray] = []
    for freq in raw_freqs:
        if any(
            np.abs(freq - _freq) <= TOLERANCE_LEVEL for _freq in unique_freqs
        ):
            continue
        unique_freqs.append(freq)

    return np.sort(np.array(unique_freqs[:n_sinusoids]))


def find_freqs_from_roots(
    coefficients: FloatArray | ComplexArray, fs: float, n_sinusoids: int
) -> FloatArray:
    """Find roots of the polynomial and estimate frequencies.

    Args:
        coefficients (FloatArray | ComplexArray):
            The polynomial coefficients.
        fs (float):
            Sampling frequency in Hz.
        n_sinusoids (int):
            Number of sinusoids.

    Returns:
        FloatArray:
            An array of estimated frequencies in Hz.
    """
    # 1. Calculate the roots of a polynomial
    try:
        roots = poly.polyroots(coefficients[::-1])
    except np.linalg.LinAlgError:
        warnings.warn("Failed to find roots of the polynomial.")
        return np.array([])

    # 2. Select the 4M roots that are closest to the unit circle
    #     Ideally, 2M candidates would be sufficient, but since some
    #     candidates may be overlooked due to noise and numerical
    #     errors, it is recommended to secure a larger number of
    #     candidates.
    sorted_indices = np.argsort(np.abs(np.abs(roots) - 1))
    closest_roots = roots[sorted_indices[: 4 * n_sinusoids]]

    # 3. Estimate normalized angular frequency from the argument of the
    #    root
    _angles = np.angle(closest_roots)
    angles = _angles[_angles >= 0]

    # 4. Convert normalized angular frequency ω [rad/sample] to physical
    #    frequency f [Hz]
    raw_freqs = angles.astype(NumpyFloat) * (fs / (2 * np.pi))

    # 5. Filter frequencies
    unique_freqs = filter_unique_freqs(raw_freqs, n_sinusoids)

    return unique_freqs


def _find_and_refine_strongest_peak(
    spectrum: FloatArray, fs: float, n_fft: int, is_real_signal: bool
) -> float:
    """Find and refine the strongest peak in an FFT spectrum.

    This helper function identifies the frequency bin with the maximum
    magnitude in the first half of a given FFT spectrum. It then applies
    parabolic interpolation to the peak and its two neighbors to
    estimate a more precise, off-grid frequency.

    Args:
        spectrum (FloatArray):
            Magnitude spectrum of a signal (the result of
            np.abs(np.fft.fft(...)))
        fs (float):
            Sampling frequency in Hz, used for frequency conversion.
        n_fft (int):
            Number of points used in the FFT calculation, required
            for correct frequency scaling.
        is_real_signal (bool):
            A flag indicating whether the original signal was
            real-valued. This determines how the spectrum is searched:
            if True, only the first half (positive frequencies) of the
            spectrum is considered.  f False, the entire shifted
            spectrum (positive and negative frequencies) is searched.

    Returns:
        float:
            The estimated frequency of the strongest peak in Hz.
    """
    if is_real_signal:
        target_spectrum = spectrum[: n_fft // 2]
        freq_grid = fftfreq(n_fft, d=1 / fs)[: n_fft // 2]
    else:
        target_spectrum = fftshift(spectrum)
        freq_grid = fftshift(fftfreq(n_fft, d=1 / fs))

    peak_idx = np.argmax(target_spectrum)
    peak_freq = freq_grid[peak_idx]
    if 0 < peak_idx < len(target_spectrum) - 1:
        y_m1, y_0, y_p1 = target_spectrum[peak_idx - 1 : peak_idx + 2]
        p = _compute_parabolic_offset(y_m1, y_0, y_p1)
        if p > 0:
            est_freq = (1 - p) * peak_freq + p * freq_grid[peak_idx + 1]
        else:
            est_freq = (1 + p) * peak_freq - p * freq_grid[peak_idx - 1]
    else:
        est_freq = peak_freq

    return float(est_freq)


def _estimate_and_subtract_component(
    signal: ComplexArray, freq: float, fs: float
) -> ComplexArray:
    """Estimate and subtract a single sinusoid from a signal.

    This function performs one step of a signal decomposition process.
    Given a signal and a single frequency, it estimates the complex
    amplitude (amplitude and phase) of the sinusoid at that frequency
    using a least-squares fit. It then synthesizes this component and
    subtracts it from the original signal, returning the residual.

    Args:
        signal (ComplexArray):
            Complex-valued input signal from which to subtract a
            component.
        freq (float):
            The frequency of the sinusoidal component to estimate and
            subtract, in Hz.
        fs (float):
            Sampling frequency in Hz.

    Returns:
        ComplexArray:
            The residual signal after the estimated component has been
            subtracted.
            Returns the original signal on failure.
    """
    t = np.arange(signal.size) / fs
    steering_vector = np.exp(2j * np.pi * freq * t).reshape(-1, 1)

    try:
        complex_amp = pinv(steering_vector) @ signal
    except LinAlgError:
        return signal

    estimated_component: ComplexArray = (
        complex_amp * steering_vector
    ).flatten()

    return signal - estimated_component


def estimate_freqs_iterative_fft(
    signal: SignalArray,
    n_peaks: int,
    fs: float,
    n_fft: int | None = None,
    *,
    is_complex: bool = False,
) -> FloatArray:
    """Estimate frequencies via an iterative interpolated FFT method.

    This method, similar to IIp-DFT, iteratively finds the strongest
    sinusoidal component, refines its frequency with interpolation,
    and subtracts it from the signal to find subsequent components.

    Args:
        signal (SignalArray):
            Input signal, which can be real or complex. The nature of
            the original signal should be indicated by the `is_complex`
            flag for correct spectral searching.
        n_peaks (int):
            Number of peaks to find and return.
        fs (float):
            Sampling frequency in Hz.
        n_fft (int, optional):
            Length of the FFT used for spectral peak finding. If None,
            it defaults to the length of the input signal. Defaults to
            None.
        is_complex (bool, optional):
            A flag indicating whether the original signal is complex-
            valued. If False (default), the search for spectral peaks
            is restricted to the positive frequency range (0 to fs/2),
            assuming the input is a real-valued signal. If True, the
            full frequency range (-fs/2 to fs/2) is searched.

    Returns:
        FloatArray: An array of estimated frequencies in Hz.
    """
    if n_fft is None:
        n_fft = signal.size

    residual_signal = signal.copy().astype(NumpyComplex)
    estimated_freqs: list[float] = []
    is_real = not is_complex

    for _ in range(n_peaks):
        if np.all(np.abs(residual_signal) < ZERO_LEVEL):
            break

        # 1. Calculate the FFT spectrum of the current signal
        spectrum = np.abs(fft(residual_signal, n=n_fft))

        # 2. Estimate the frequency of the strongest peak
        est_freq = _find_and_refine_strongest_peak(
            spectrum, fs, n_fft, is_real_signal=is_real
        )
        if is_real and est_freq < 0:
            continue
        estimated_freqs.append(est_freq)

        # 3. Remove the estimated component from the residual signal
        residual_signal = _estimate_and_subtract_component(
            residual_signal, est_freq, fs
        )

    return np.sort(np.array(estimated_freqs))
