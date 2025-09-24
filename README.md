# High-Resolution Parameter Estimation for Sinusoidal Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides Python implementations of high-resolution parameter estimation algorithms for sinusoidal signals including **MUSIC (Spectral/Root)** and **ESPRIT**. The project is structured with an object-oriented approach, emphasizing code clarity, reusability, and educational value.

This work is inspired by the foundational papers in subspace-based signal processing and aims to provide a practical and understandable guide to these powerful techniques.

## Features

- **High-Resolution Algorithms**: Implements spectral estimation techniques that surpass the resolution limits of the classical Fast Fourier Transform (FFT).
- **Multiple Methods Implemented**: A comprehensive suite of high-resolution algorithms is provided:
  - **MUSIC (Spectral, Root & FAST)**: A classic high-resolution method based on the orthogonality of signal and noise subspaces.
    - The **FAST MUSIC** variant provides a computationally efficient implementation for (quasi-)periodic signals by replacing the expensive EVD with an FFT. 
  - **Min-Norm (Spectral & Root)**: A variant of MUSIC that can reduce computational cost by using a single, optimized vector from the noise subspace.
  - **ESPRIT (Standard & Unitary)**: A computationally efficient method that estimates parameters directly without spectral search by exploiting rotational invariance.
    - The **Standard ESPRIT** (LS/TLS) provides a direct algebraic solution in the complex domain.
    - The **Unitary ESPRIT** (LS/TLS) variant transforms the problem into the real domain, significantly reducing computational complexity.
  - **HOYW**: A robust method based on the autocorrelation function and an AR model of the signal, enhanced with SVD-based rank truncation.
- **Full Parameter Estimation**: Not just frequencies, but also amplitudes and phases are estimated using a subsequent least-squares fit.
- **Object-Oriented Design**: Algorithms are encapsulated in clear, reusable classes with a consistent API, promoting clean code and extensibility.
- **Enhanced Accuracy Options**: Includes advanced techniques like **Forward-Backward Averaging** (via Mixins) and **Total Least Squares (TLS)** versions for most algorithms to improve performance in noisy conditions.
- **Clean and Type-Hinted Code**: Fully type-hinted with `mypy` validation. Code quality is enforced by a combination of `ruff` (for speed and formatting) and `pylint` (for in-depth analysis), ensuring high standards of maintainability.
- **Demonstration Script**: Includes a flexible command-line interface (`main.py`) to easily run experiments and compare the performance of different algorithms and their variants.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/tam17aki/music-esprit-python.git
    cd music-esprit-python
    ```

2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:

     This is the recommended way to install the package for development or running examples. It installs the necessary dependencies and makes the package available throughout your environment.
    ```bash
    pip install -e .
    ```
    *(Note: This command reads the `pyproject.toml` file to understand how to install the package.)*

## Usage

After installation, you can run the provided example scripts to see the analyzers in action. These scripts are located in the `examples/` directory and serve as a starting point for your own experiments.

### Basic Example

To run a comparative analysis of MUSIC and ESPRIT with default parameters (a signal with three sinusoids at 440, 460, and 480 Hz):

```bash
python examples/run_comparison.py
```

#### Example Output
Running the command above will produce an output similar to this:

```
--- Experiment Setup ---
Sampling Frequency: 44100.0 Hz
Signal Duration:    100 ms
SNR:                30.0 dB
True Frequencies:   [440. 460. 480.] Hz
True Amplitudes:    [1.08607452 1.26224867 1.02368536]
True Phases:        [-1.84613126 -2.56550767  1.29983362] rad

--- Running Spectral MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  N Grids: 16384
Elapsed Time: 0.7337 seconds

--- Estimation Results ---
Est Frequencies: [440.21491002 460.26970412 479.18628111] Hz
Est Amplitudes:  [1.05378802 1.23946073 0.99548543]
Est Phases:      [-1.90415797 -2.66810899  1.56569674] rad

--- Estimation Errors ---
Freq Errors:  [ 0.21491002  0.26970412 -0.81371889] Hz
Amp Errors:   [-0.0322865  -0.02278794 -0.02819992]
Phase Errors: [-0.05802671 -0.10260131  0.26586312] rad


--- Running Root MUSIC ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
Elapsed Time: 15.5962 seconds

--- Estimation Results ---
Est Frequencies: [440.00114041 460.01159413 479.99561256] Hz
Est Amplitudes:  [1.08709193 1.26275134 1.02308641]
Est Phases:      [-1.84674599 -2.57022537  1.3015119 ] rad

--- Estimation Errors ---
Freq Errors:  [ 0.00114041  0.01159413 -0.00438744] Hz
Amp Errors:   [ 0.00101742  0.00050267 -0.00059895]
Phase Errors: [-0.00061473 -0.00471769  0.00167828] rad


--- Running ESPRIT (LS) ---
Analyzer Parameters:
  Subspace Ratio: 0.3333
  Solver: LSEspritSolver
Elapsed Time: 0.4664 seconds

--- Estimation Results ---
Est Frequencies: [439.98992195 460.01069727 479.99639225] Hz
Est Amplitudes:  [1.08715901 1.26232343 1.02343474]
Est Phases:      [-1.8432428  -2.57024044  1.30123083] rad

--- Estimation Errors ---
Freq Errors:  [-0.01007805  0.01069727 -0.00360775] Hz
Amp Errors:   [ 1.08449626e-03  7.47562613e-05 -2.50621360e-04]
Phase Errors: [ 0.00288846 -0.00473276  0.00139721] rad
```

(Note: The exact values for amplitudes, phases, and errors will vary due to their random generation.)

### Running Focused Comparisons

For more specific comparisons, you can run other example scripts:

- `examples/compare_music_variants.py`:<br>
This script focuses exclusively on the MUSIC family, comparing the performance of `SpectralMusicAnalyzer`, `RootMusicAnalyzer`, `SpectralMinNormAnalyzer`, and `RootMinNormAnalyzer`.

```bash
python examples/compare_music_variants.py
```

- `examples/compare_esprit_variants.py`:<br>
This script is dedicated to the ESPRIT family. It compares `StandardEspritAnalyzer` against the computationally efficient `UnitaryEspritAnalyzer` (with both LS and TLS solvers).

```bash
python examples/compare_esprit_variants.py
```

You can customize the experiments by modifying these scripts or by using the command-line arguments they provide.

### Command-Line Options

You can customize the experiment via command-line arguments.

```bash
python examples/run_comparison.py --freqs_true 440 445 450 --snr_db 25 --duration 0.8 --complex
```

To see all available options, run:

```bash
python examples/run_comparison.py --help
```

| Argument| Description | Default |
| :-------- | :-------- | :-------- |
|`--fs`| Sampling frequency in Hz.| 44100.0 |
| `--duration` | Signal duration in seconds. | 0.1|
|`--snr_db` | Signal-to-Noise Ratio in dB. | 30.0|
| `--freqs_true`  | A list of true frequencies in Hz. | 440 460 480|
| `--amp_range` | The range for random amplitude generation. | 0.5 1.5|
| `--subspace_ratio` | The ratio of the subspace dimension to the signal length.<br> Must be in (0, 0.5].| 1/3|
| `--complex` | If specified, generate a complex-valued signal instead of <br> a real-valued one.| False (Flag)|
| `--n_grids` | Number of grid points for Spectral MUSIC and Spectral Min-Norm. | 16384|

## Project Structure

This project is organized into a modular, object-oriented structure to promote clarity, reusability, and separation of concerns. The core logic is built upon a hierarchical class system.

- **`main.py`**: The main entry point to run demonstrations. It orchestrates the setup, execution, and result presentation of the analysis.
- **`analyzers/`**: A package containing the core implementations of the signal processing algorithms, structured as a class hierarchy.
  - `base.py`: Defines `AnalyzerBase`, the top-level abstract base class for all parametric estimation methods. It contains the common logic for the analysis workflow, such as the `fit` method template, subsequent amplitude/phase estimation, and result properties.
  - `models.py`: Defines `TypedDict` models (e.g., `AnalyzerParameters`) used for structuring and type-hinting the dictionaries of hyperparameters passed to and returned by the analyzers.
  - **`music/`**: A sub-package dedicated to the MUSIC algorithm and its variants.
    - `base.py`: Defines `MusicAnalyzerBase`, an intermediate abstract class for all MUSIC variants. It inherits from `AnalyzerBase` and adds MUSIC-specific logic, like the estimation of the noise subspace.
    - `spectral.py`: Implements `SpectralMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via spectral peak-picking.
    - `root.py`: Implements `RootMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies via polynomial rooting.
    - `fast.py`: Implements `FastMusicAnalyzer` (inheriting from `MusicAnalyzerBase`), which estimates frequencies for periodic signals by analyzing the peaks of the ACF's power spectrum and computing a closed-form pseudospectrum.
  - **`esprit/`**: A sub-package dedicated to the ESPRIT algorithm and its variants, including the computationally efficient Unitary ESPRIT.
    - `base.py`:  Defines `EspritAnalyzerBase`, an intermediate abstract class for ESPRIT-based methods. It inherits from `AnalyzerBase`, and adds ESPRIT-specific logic, like the estimation of the signal subspace.
    - `standard.py`: Implements `StandardEspritAnalyzer` for the classic, complex-valued ESPRIT algorithm.
    - `unitary.py`: Implements `UnitaryEspritAnalyzer`, which operates entirely on real-valued matrices for reduced computational load and improved accuracy.
    - `solvers.py`: Defines a set of solver classes that encapsulate the specific mathematical procedures for solving the ESPRIT core equations. This demonstrates the Strategy design pattern, allowing different numerical methods (LS, TLS, Unitary LS/TLS) to be flexibly injected into the analyzers.
  - **`minnorm/`**: A sub-package for Min-Norm algorithm variants.
    - `base.py`: Defines `MinNormAnalyzerBase`, containing the core logic for computing the minimum norm vector.
    - `spectral.py`: Implements `SpectralMinNormAnalyzer`, which estimates frequencies via spectral peak-picking.
    - `root.py`: Implements `RootMinNormAnalyzer`, which estimates frequencies via polynomial rooting.
  - **`hoyw/`**: A sub-package for the Higher-Order Yule-Walker (HOYW) method.
     - `hoyw.py`: Implements `HoywAnalyzer`, which directly inherits from `AnalyzerBase`. It estimates frequencies by solving the HOYW equations and subsequent finding the polynomial roots.
- **`mixins/`**: A package for providing optional enhancements to the analyzer classes through multiple inheritance.
  - `covariance.py`: Contains the `ForwardBackwardMixin` to add Forward-Backward averaging capability.
- **`utils/`**: A package for reusable helper modules and data structures that are decoupled from the specific analyzer implementations.
  - `data_models.py`: Defines the core `dataclass` models for the project.
    - `ExperimentConfig`: Encapsulates all parameters for a simulation run (e.g., SNR, duration), defining the "world" in which the signals exist.
    - `SinusoidParameters`: Represents the ground truth or estimated parameters of a signal, serving as the data "payload" that is generated and analyzed.
  - `signal_generator.py`: Provides functions for synthesizing test signals.
- **`cli.py`**: A module dedicated to the Command-Line Interface. It handles argument parsing and the formatting of results for display.
-  **`examples/`**: A directory containing example scripts that demonstrate how to use the library.
    - `run_comparison.py`: The main demonstration script that runs a comparative analysis of Spectral/Root MUSIC and ESPRIT algorithm. It can be used as a starting point for your own experiments.
    - `compare_music_variants.py`: The demonstration script that runs a comparative analysis of Spectral and Root MUSIC algorithm, including their Forward-Backward enhanced versions.
    - `compare_esprit_variants.py`: The demonstration script that runs a comparative analysis of Standard ESPRIT (LS/TLS) and  Unitary ESPRIT (LS/TLS) algorithm.

This layered design allows for maximum code reuse and easy extension.

## Architecture Overview

The project is built upon a flexible and extensible object-oriented architecture. The core of the library is a hierarchical system of analyzer classes, designed to maximize code reuse and clearly separate concerns.

The class diagram below illustrates the main **inheritance relationships** between the analyzer classes.

![Simple Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/simple_class_diagram.png)
*<div align="center">Fig. 1: Primary inheritance hierarchy of the analyzer classes</div>*

As shown, all analyzers inherit from a common `AnalyzerBase`, ensuring a consistent API. Specialized abstract classes like `MusicAnalyzerBase` and `EspritAnalyzerBase` group together logic common to each algorithm family.

### Key Design Patterns

Beyond this basic inheritance, the architecture leverages several key design patterns to add features and flexibility in a modular way.

-   **Strategy Pattern for Decoupling**: The ESPRIT algorithm's core numerical procedure is decoupled from the main analyzer class. The analyzers delegate this task to separate **`Solver`** objects (`LSEspritSolver`, `TLSUnitaryEspritSolver`, etc.), allowing different numerical solution strategies to be flexibly "plugged in."

-   **Mixin Classes for Feature Enhancement**: Optional features, such as Forward-Backward averaging, are added to concrete analyzers using **Mixin classes** (e.g., `ForwardBackwardMixin`). This allows for functionality to be added via composition, avoiding a rigid and deep inheritance tree.

-   **Structured Data Modeling**: The project heavily utilizes Python's typing features to create robust and self-documenting data structures.
    - **Analyzer's Public API (`get_params`)**: The `.get_params()` method returns a `TypedDict` model (`AnalyzerParameters`) instead of a plain dictionary. This provides a clear, type-safe structure for reporting the analyzers' hyperparameters.
    - **Analyzer's State (`est_params`)**: The estimation results are stored in an immutable `dataclass` object, `SinusoidParameters`. This encapsulates the output data (frequencies, amplitudes, phases) and ensures that the results, once computed, cannot be accidentally modified.

The complete architecture, including these mixin and composition relationships, is shown in the detailed class diagram below for those interested in the full implementation details.

![Complete Class Diagram](https://github.com/tam17aki/music-esprit-python/blob/main/docs/images/complete_class_diagram.png)
*<div align="center">Fig. 2: Detailed class diagram including Mixins, Solvers, and Data Models</div>* 

## Theoretical Background

The implemented methods are **model-based** high-resolution techniques that estimate sinusoidal parameters by fitting the observed signal to a predefined mathematical model. This approach allows for performance far exceeding that of traditional non-parametric methods like the FFT.

Two main families of models are explored in this project:

1.  **Subspace Models (MUSIC, ESPRIT, Min-Norm):** These methods model the signal's covariance matrix as having a low-rank signal component embedded in noise. They exploit the geometric properties of the signal and noise subspaces, which are obtained via eigenvalue decomposition. For a detailed, step-by-step walkthrough of the MUSIC algorithm, please see the web version or download the PDF:
     - [Web Version (Markdown)](docs/theory/music_theory.md)
     - [Printable Version (PDF)](docs/theory/music_theory.pdf)
2.  **Autoregressive (AR) Models (HOYW):** This approach models the signal as the output of a linear time-invariant system driven by white noise. Frequencies are estimated from the roots of the AR model's characteristic polynomial, whose coefficients are found from the signal's autocorrelation sequence.

For a deeper dive into the theory, please refer to the papers [1-3] for Spectral MUSIC, [4] for Root MUSIC, [5] for FAST MUSIC, [6] for Min-Norm, [7] for ESPRIT, [8] for Unitary ESPRIT, and [9] for HOYW. 
The comprehensive textbook [10] provides detailed mathematical derivations and analyses of these methods and many other advanced signal processing techniques.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/tam17aki/music-esprit-python/blob/main/LICENSE) file for details.

## References
[1] Schmidt, R. O. (1979). “Multiple emitter location and signal parameter estimation,” in Proc. RADC, Spectral Estimation Workshop, Rome, NY, pp. 243–258.

[2] Bienvenu, G. (1979). “Influence of the spatial coherence of the background noise on high resolution passive methods,” in Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, Washington, DC, pp. 306–309.

[3] R.O. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Trans. Antennas and Propagat., vol. AP-34, no. 3, pp. 276-280, 1986.

[4] A. Barabell, "Improving the resolution performance of eigenstructure-based direction-finding algorithms," in IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 8, pp. 336-339, 1983.

[5] O. Das, J. S. Abel, J. O. Smith III, "FAST MUSIC - An Efficient Implementation of The Music Algorithm For Frequency Estimation of Approximately Periodic Signals," International Conference on Digital Audio Effects (DAFx), vol.21, pp.342-348, 2018.

[6] R. Kumaresan and D. W. Tufts, "Estimating the Angles of Arrival of Multiple Plane Waves," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-19, no. 1, pp. 134-139, 1983.

[7] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984-995, 1989.

[8] M. Haardt and J. A. Nossek, "Unitary ESPRIT: how to obtain increased estimation accuracy with a reduced computational burden," in IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1232-1242, 1995.

[9] P. Stoica, T. Soderstrom and F. Ti, "Asymptotic properties of the high-order Yule-Walker estimates of sinusoidal frequencies," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 11, pp. 1721-1734, 1989.

[10] P. Stoica and R. Moses, "Spectral Analysis of Signals," Pearson Prentice Hall, 2005.
