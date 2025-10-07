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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        fs: float,
        n_sinusoids: int,
        *,
        n_newton_steps: int = 1,
        n_cyclic_rounds: int = 1,
        convergence_threshold: float = 1e-6,
    ):
        """Initialize the NOMP analyzer.

        Args:
            fs (float): Sampling frequency in Hz.
            n_sinusoids (int): Number of sinusoids to estimate.
            n_newton_steps (int, optional): Number of Newton refinement
                steps for each new component (`R_s` in the paper).
                Defaults to 1.
            n_cyclic_rounds (int, optional): Maximum number of full
                cyclic refinement rounds after each new component (`R_c`
                in the paper). Defaults to 1.
            convergence_threshold (float, optional): Threshold for
                stopping the cyclic refinement. If the relative energy
                improvement drops below this value, the loop terminates.
                Set to 0 to disable. Defaults to 1e-6.
        """
        super().__init__(fs, n_sinusoids)
        self.n_newton_steps = n_newton_steps
        self.n_cyclic_rounds = n_cyclic_rounds
        self.convergence_threshold = convergence_threshold

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
            residual = self._compute_residual_signal(
                estimated_freqs, complex_signal
            )

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
        refined_freqs = np.copy(current_freqs)
        prev_residual_energy = np.inf
        converged = False

        # Iterate up to the maximum number of rounds specified.
        for cycle in range(self.n_cyclic_rounds):
            # Iterate through each frequency in the current estimated
            # set.
            freqs_before_update = np.copy(refined_freqs)
            for i, freq_to_refine in enumerate(freqs_before_update):
                # Form a temporary residual for the component to be
                # refined.
                temp_residual = self._get_residual_for_single_component(
                    original_signal, freqs_before_update, i
                )

                # Refine the target frequency against this temporary
                # residual.
                updated_freq = self._newton_refinement_step(
                    temp_residual, freq_to_refine
                )
                refined_freqs[i] = updated_freq

            # Check for convergence after the second cycle.
            if self.convergence_threshold > 0 and cycle > 0:
                prev_residual_energy = self._compute_energy(
                    freqs_before_update, original_signal
                )
                current_residual_energy = self._compute_energy(
                    refined_freqs, original_signal
                )
                if prev_residual_energy == np.inf:
                    warnings.warn(
                        "Cyclic refinement stopped due to energy calculation "
                        + "failure."
                    )
                    break

                # Calculate relative improvement in residual energy.
                relative_improvement = abs(
                    prev_residual_energy - current_residual_energy
                ) / (prev_residual_energy + ZERO_LEVEL)

                # Stop if improvement is below the threshold.
                if relative_improvement < self.convergence_threshold:
                    print(
                        "  (Cyclic refinement converged after"
                        + f" {cycle + 1} rounds)"
                    )
                    converged = True
                    break

        if not converged and self.convergence_threshold > 0:
            print(
                "  (Cyclic refinement stopped after reaching max "
                + f"{self.n_cyclic_rounds} rounds)"
            )

        return refined_freqs

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

    def _get_residual_for_single_component(
        self,
        original_signal: ComplexArray,
        all_freqs: FloatArray,
        index_to_exclude: int,
    ) -> ComplexArray:
        """Compute the residual signal for refining a single component.

        This helper function is used during the cyclic refinement phase.
        It computes a temporary residual by performing a least-squares
        fit of all components *except* for the one at the specified
        index, and subtracting this model from the original signal.

        Args:
            original_signal (ComplexArray):
                The original, unmodified input signal.
            all_freqs (FloatArray):
                An array of all current frequency estimates.
            index_to_exclude (int):
                The index of the frequency component to
                exclude from the model.

        Returns:
            ComplexArray:
                The temporary residual signal.
        """
        other_freqs = np.delete(all_freqs, index_to_exclude)
        if other_freqs.size == 0:
            return original_signal
        residual = self._compute_residual_signal(other_freqs, original_signal)
        return residual

    def _compute_energy(
        self, freqs: FloatArray, signal: ComplexArray
    ) -> float:
        """Compute the energy for a given frequency set.

        This method uses the core LS fitting logic to compute the
        residual signal for a given set of frequencies and then
        calculates its squared L2-norm (energy).

        Args:
            freqs (FloatArray):
                The set of frequencies to build the model from.
            signal (ComplexArray):
                The signal to fit against.

        Returns:
            float:
                The energy of the residual signal. On the residual
                calculation failulre, returns an infinite energy
                (np.inf).
        """
        residual = self._compute_residual_signal(freqs, signal)

        # If the residual calculation failed, return an infinite energy.
        if residual is signal:
            return np.inf
        energy = float(np.real(np.vdot(residual, residual)))
        return energy

    def _compute_residual_signal(
        self, freqs: FloatArray, signal: ComplexArray
    ) -> ComplexArray:
        """Compute the residual signal from frequencies via LS fit.

        This is the central least-squares (LS) fitting logic for NOMP.
        Given a set of frequencies and a signal, it constructs a
        Vandermonde matrix, solves for the optimal complex amplitudes,
        and computes the corresponding residual signal.

        Args:
            freqs (FloatArray):
                The set of frequencies to build the model from.
            signal (ComplexArray):
                The signal to fit against.

        Returns:
            ComplexArray:
                The residual signal after subtraction of the fitted
                model. On calculation failure, returns the original
                signal.
        """
        n_samples = signal.size
        vandermonde = self._build_vandermonde_matrix(freqs, n_samples, self.fs)

        try:
            amps = pinv(vandermonde) @ signal
            residual = signal - (vandermonde @ amps)
            return residual
        except LinAlgError:
            warnings.warn("LS fit failed in _compute_residual_signal.")
            return signal

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
        params["convergence_threshold"] = self.convergence_threshold
        return params
