# -*- coding: utf-8 -*-
"""Defines NompAnalyzer class for the NOMP algorithm.

This module implements the Newtonized Orthogonal Matching Pursuit (NOMP)
algorithm, a greedy iterative method for high-resolution sinusoidal parameter
estimation.

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
from typing import final, override

import numpy as np
from scipy.linalg import LinAlgError, pinv

from utils.data_models import (
    ComplexArray,
    FloatArray,
    NumpyComplex,
    SignalArray,
)

from .._common import ZERO_LEVEL
from ..base import AnalyzerBase
from ..models import AnalyzerParameters


@final
class NompAnalyzer(AnalyzerBase):
    """Analyzes sinusoids using Newtonized Orthogonal Matching Pursuit.

    NOMP is a greedy iterative algorithm that enhances Orthogonal
    Matching Pursuit (OMP) with two key refinement stages:
    1.  A Newton-Raphson step to refine each newly detected frequency
        estimate off the DFT grid (Single Refinement).
    2.  A cyclic process to re-evaluate all previously found
        frequencies in light of the new component (Cyclic Refinement).
    3.  A final least-squares fit of all component amplitudes against
        the original signal after each new component is added.

    This feedback mechanism allows the algorithm to correct earlier
    estimates and achieve higher accuracy than simpler iterative methods
    like RELAX, especially when components interfere with each other.

    Reference:
        B. Mamandipoor, et al., "Newtonized Orthogonal Matching Pursuit:
        Frequency Estimation Over the Continuum," in IEEE Transactions
        on Signal Processing, vol. 64, no. 19, pp. 5066-5081, 2016.
    """

    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        n_newton_steps: int = 1,
        n_cyclic_rounds: int = 1,
    ):
        """Initialize the NOMP analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids to estimate.
            n_newton_steps (int, optional): Number of Newton refinement
                steps to apply for each new component (`R_s` in the
                paper). Defaults to 1.
            n_cyclic_rounds (int, optional): Number of full cyclic
                refinement rounds to perform after each new component
                (`R_c` in the paper). Defaults to 1.
        """
        super().__init__(fs, n_sinusoids)
        self.n_newton_steps: int = n_newton_steps
        self.n_cyclic_rounds: int = n_cyclic_rounds

    @override
    def _estimate_frequencies(self, signal: SignalArray) -> FloatArray:
        """Estimate frequencies using the NOMP algorithm workflow.

        This method orchestrates the main loop of the NOMP algorithm. In
        each iteration, it performs the following core steps by calling
        specialized helper methods:
        1.  Identifies and refines a new frequency component from the
            current residual signal.
        2.  Performs cyclic refinement on the entire set of currently
            estimated frequencies.
        3.  Updates the amplitudes of all components via a final least-
            squares fit and computes the new residual for the next
            iteration.

        Args:
            signal (SignalArray): The input signal to be analyzed.

        Returns:
            FloatArray: A sorted array of the final estimated
                frequencies in Hz.
        """
        complex_signal = signal.astype(NumpyComplex)

        estimated_freqs = np.array([], dtype=np.float64)
        residual = np.copy(complex_signal)

        # Iterate to find n_sinusoids components
        for _ in range(self.n_sinusoids):
            # Step 1 & 2 of NOMP: Find a new component on the current
            # residual and refine it with Newton's method.
            new_freq = self._identify_and_refine_new_component(
                residual, is_real_signal=np.isrealobj(signal)
            )
            if new_freq is None:
                break
            estimated_freqs = np.append(estimated_freqs, new_freq)

            # Step 3 of NOMP: With the new component added, re-refine
            # all previously found frequencies in a cyclic manner.
            estimated_freqs = self._perform_cyclic_refinement(
                complex_signal, estimated_freqs
            )

            # Step 4 & 5 of NOMP: Update all amplitudes with a final LS
            # fit and compute the new residual for the next iteration.
            residual = self._update_residual(complex_signal, estimated_freqs)

        return np.sort(estimated_freqs)

    def _identify_and_refine_new_component(
        self, residual: ComplexArray, is_real_signal: bool
    ) -> float | None:
        """Identify a new component and perform single refinement.

        This method corresponds to the "IDENTIFY" and "SINGLE
        REFINEMENT" steps of the NOMP algorithm. It finds the strongest
        peak in the DFT spectrum of the current residual signal and
        refines its frequency using Newton's method.

        Args:
            residual (ComplexArray):
                The current residual signal to search within.
            is_real_signal (bool):
                A flag indicating if the original signal was
                real-valued, used to determine the spectral search
                range.

        Returns:
            float | None:
                The refined frequency of the newly identified component
                in Hz, or None if no peak is found.
        """
        n_samples = residual.size
        dft_residual = np.fft.fft(residual)

        # Coarsely identify the strongest frequency component by finding
        # the peak of the DFT spectrum (the "IDENTIFY" step).
        if is_real_signal:
            # For real signals, search only positive frequencies.
            search_space = np.abs(dft_residual[1 : n_samples // 2])
            if search_space.size == 0:
                return None
            k_c = int(np.argmax(search_space)) + 1
            coarse_freq = (k_c / n_samples) * self.fs
        else:
            # For complex signals, search the full frequency range.
            shifted_spectrum = np.abs(np.fft.fftshift(dft_residual))
            peak_idx_shifted = int(np.argmax(shifted_spectrum))
            shifted_freq_grid = np.fft.fftshift(
                np.fft.fftfreq(n_samples, d=1 / self.fs)
            )
            coarse_freq = shifted_freq_grid[peak_idx_shifted]

        # Refine the coarse estimate using Newton's method for a
        # specified number of steps (the "SINGLE REFINEMENT" step).
        refined_freq = coarse_freq
        for _ in range(self.n_newton_steps):
            refined_freq = self._newton_refinement_step(residual, refined_freq)

        return refined_freq

    def _perform_cyclic_refinement(
        self, original_signal: ComplexArray, current_freqs: FloatArray
    ) -> FloatArray:
        """Perform cyclic refinement on all estimated frequencies.

        This method corresponds to the "CYCLIC REFINEMENT" step of the
        NOMP algorithm. It iterates through all currently estimated
        frequencies, re-refining each one using Newton's method against
        a residual formed by subtracting all other components from the
        original signal. This process is repeated for a specified number
        of rounds (`n_cyclic_rounds`).

        Args:
            original_signal (ComplexArray):
                The original, unmodified input signal.
            current_freqs (FloatArray):
                An array of all frequency estimates found so far.

        Returns:
            FloatArray:
                An array of updated frequency estimates after cyclic
                refinement.
        """
        n_samples = original_signal.size
        refined_freqs = np.copy(current_freqs)

        # Repeat the entire cycle of refinements for n_cyclic_rounds.
        for _ in range(self.n_cyclic_rounds):
            # Iterate through each frequency in the current estimated
            # set.
            for i, freq_to_refine in enumerate(refined_freqs):
                # Form a temporary residual by subtracting all OTHER
                # components from the ORIGINAL signal.
                other_freqs = np.delete(refined_freqs, i)
                if other_freqs.size > 0:
                    vandermonde = self._build_vandermonde_matrix(
                        other_freqs, n_samples, self.fs
                    )
                    try:
                        other_amps = pinv(vandermonde) @ original_signal
                    except LinAlgError:
                        warnings.warn(
                            "pinv failed in cyclic refinement. "
                            + "Skipping update."
                        )
                        continue
                    other_components = vandermonde @ other_amps
                    temp_residual = original_signal - other_components
                else:
                    # If this is the only component, the residual is the
                    # original signal itself.
                    temp_residual = original_signal

                # Refine the target frequency against this temporary
                # residual.
                updated_freq = self._newton_refinement_step(
                    temp_residual, freq_to_refine
                )
                refined_freqs[i] = updated_freq

        return refined_freqs

    def _update_residual(
        self, original_signal: ComplexArray, estimated_freqs: FloatArray
    ) -> ComplexArray:
        """Update all amplitudes via LS and compute the final residual.

        This method corresponds to the "UPDATE" step of the NOMP
        algorithm. It performs a final least-squares (LS) fit of all
        currently estimated frequency components against the original
        signal to obtain optimal amplitudes. It then computes and
        returns the final residual signal for the next iteration.

        Args:
            original_signal (ComplexArray):
                The original, unmodified input signal.
            estimated_freqs (FloatArray):
                The latest array of refined frequency estimates.

        Returns:
            ComplexArray: The new residual signal after subtracting all
                optimally fitted components.
        """
        n_samples = original_signal.size

        # Construct the Vandermonde matrix from all current frequency
        # estimates.
        vandermonde_all = self._build_vandermonde_matrix(
            estimated_freqs, n_samples, self.fs
        )
        try:
            # Solve for all amplitudes simultaneously via a
            # least-squares fit against the original signal.
            all_amps = pinv(vandermonde_all) @ original_signal
        except LinAlgError:
            warnings.warn("Final residual update failed due to LinAlgError.")
            return original_signal

        # Reconstruct the signal model with the refined frequencies and
        # amplitudes.
        all_components = vandermonde_all @ all_amps

        # Return the final residual for the next main loop iteration.
        return original_signal - all_components

    def _newton_refinement_step(
        self, target_signal: ComplexArray, current_freq: float
    ) -> float:
        """Perform a single Newton-Raphson refinement step.

        This method computes the first and second derivatives of the
        cost function with respect to the frequency and applies one
        Newton's method update `w_new = w - Ṡ / S̈`.

        Args:
            target_signal (ComplexArray):
                The signal (or residual) to fit against.
            current_freq (float):
                The current frequency estimate in Hz.

        Returns:
            float:
                The refined frequency estimate in Hz.
        """
        n = target_signal.size
        t = np.arange(n)
        w = (2 * np.pi * current_freq) / self.fs  # Normalized angular freq

        # --- Calculate steering vector and its two derivatives ---
        a = np.exp(1j * w * t)
        a_dot = 1j * t * a
        a_ddot = -(t**2) * a

        # --- Calculate intermediate values for the derivatives ---
        # Complex amplitude `g` for the current frequency component
        g = np.vdot(a, target_signal) / n
        r = target_signal - g * a  # Residual w.r.t. current estimate

        # --- Compute derivatives of the cost function (Eqs. 7 & 8) ---
        # First derivative Ṡ
        s_dot = np.real(np.vdot(r, g * a_dot))
        # Second derivative S̈
        s_ddot = np.real(np.vdot(r, g * a_ddot)) - (np.abs(g) ** 2) * np.real(
            np.vdot(a_dot, a_dot)
        )

        # --- Apply Newton's update rule ---
        # Refinement Acceptance Condition (RAC): update only if the cost
        # function is locally concave (i.e., we are near a maximum).
        if s_ddot >= 0 or abs(s_ddot) < ZERO_LEVEL:
            return current_freq  # No update

        # Newton's method update rule for the angular frequency
        w_new = w - s_dot / s_ddot

        # Convert back to physical frequency in Hz
        return float(w_new * self.fs / (2 * np.pi))

    @override
    def get_params(self) -> AnalyzerParameters:
        """Return the analyzer's hyperparameters.

        Extends the base implementation to include NOMP-specific
        parameters like the number of Newton and cyclic refinement
        steps.
        """
        params = super().get_params()
        params.pop("subspace_ratio", None)
        params["n_newton_steps"] = self.n_newton_steps
        params["n_cyclic_rounds"] = self.n_cyclic_rounds
        return params
