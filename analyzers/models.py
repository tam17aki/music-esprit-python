# -*- coding: utf-8 -*-
"""Defines TypedDict models for structuring analyzer parameters and results.

This module contains the type definitions that describe the shape of data
objects, such as hyperparameter dictionaries, used across the different
analyzer classes.

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

from typing import NotRequired, TypedDict


class AnalyzerParameters(TypedDict):
    """A TypedDict model for the dictionary returned by `AnalyzerBase.get_params()`.

    This structure defines the set of all possible hyperparameters that can be
    reported by any analyzer instance in this library. Each key may or may not
    be present depending on the specific analyzer class.
    """

    subspace_ratio: float
    """Ratio of the subspace dimension to the signal length (L/N)."""

    solver: NotRequired[str]
    """The name of the solver class used (e.g., 'LSEspritSolver')."""

    n_grids: NotRequired[int]
    """Number of grid points for spectral search methods."""

    min_freq_period: NotRequired[float]
    """The minimum frequency in Hz for FAST MUSIC method."""

    ar_order: NotRequired[int]
    """The order of the  AutoRegressive (AR) model for HOYW method."""
